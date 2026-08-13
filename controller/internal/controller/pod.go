package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/yaml"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Time-slicer contract, mirrored from src/accel_timeslicer/workload.py. A pod
// label so other pods can discover the workload, an env var so the process
// itself knows which group and owner it belongs to. All of them must agree.
const (
	timeSliceEnabledLabel = "accel-timeslicer"
	timeSliceGroupLabel   = "timeslice.io/group"
	timeSliceOwnerLabel   = "timeslice.io/owner"
	timeSliceJobIDLabel   = "timeslice.io/job-id"
	timeSliceJobIDEnv     = "OPEN_RL_TIME_SLICE_JOB_ID"
	timeSliceGroupEnv     = "OPEN_RL_TIME_SLICE_GROUP"
	timeSliceOwnerEnv     = "OPEN_RL_TIME_SLICE_OWNER"
)

// The name the pod and the claim agree to call the allocation.
const podClaimName = "gpu"

// labelUnsafe matches everything a DNS-1123 label value may not contain.
var labelUnsafe = regexp.MustCompile(`[^a-z0-9-]+`)

// sanitizeLabel reduces an arbitrary id to something usable as a label value.
// Empty in, empty out.
func sanitizeLabel(value string) string {
	cleaned := strings.Trim(labelUnsafe.ReplaceAllString(strings.ToLower(value), "-"), "-")
	if len(cleaned) > 63 {
		cleaned = strings.TrimRight(cleaned[:63], "-")
	}
	return cleaned
}

// workerPodName derives the pod from the CR name -- the one identity
// Kubernetes already guarantees unique. A truncated name keeps a hash of the
// full one so two long names cannot collide.
func workerPodName(worker *openrlv1alpha1.OpenRLWorker) string {
	name := "orw-" + worker.Name
	if len(name) <= 253 {
		return name
	}
	sum := sha256.Sum256([]byte(name))
	return name[:245] + "-" + hex.EncodeToString(sum[:])[:7]
}

// claimNameFor derives the claim from the worker's UID: unique per
// incarnation, so a recreated worker can never collide with -- or blindly
// adopt -- its predecessor's claim, while repeated reconciles of one
// incarnation converge on one name. (The predecessor's claim stays reusable
// the honest way: once allocated, SelectClaim can join it with real checks.)
// The worker's name rides along for operators, truncated because the claim
// name travels as a label value (the pod's time-slice group, max 63).
func claimNameFor(worker *openrlv1alpha1.OpenRLWorker) string {
	name := worker.Name
	if len(name) > 48 {
		name = strings.TrimRight(name[:48], "-.")
	}
	uid := string(worker.UID)
	if uid == "" {
		// Objects always carry a UID in a real cluster; bare fixtures don't.
		return "claim-" + name
	}
	if len(uid) > 8 {
		uid = uid[:8]
	}
	return "claim-" + name + "-" + uid
}

// buildClaim builds a ResourceClaim: the device count we decided, plus CEL
// bounds on per-device memory. The floor is the worker's share; the ceiling
// is the device size the claim was priced against -- without it, DRA could
// satisfy an L4-sized claim with an H100 and strand a later big worker.
// Deliberately no node selector: which node satisfies this is
// kube-scheduler's decision, steered by the pod.
func (r *OpenRLWorkerReconciler) buildClaim(worker *openrlv1alpha1.OpenRLWorker, claim *placement.Claim, perDeviceBytes, deviceMemoryBytes int64) *resourcev1.ResourceClaim {
	// No role or owner label: which workers sit on a claim is rebuilt from
	// the workers that reference it. Count and device memory are the claim's
	// shape contract, checked when a create collides with an existing claim.
	labels := map[string]string{
		LabelManaged:      "true",
		LabelAccelCount:   fmt.Sprint(claim.DeviceCount),
		LabelDeviceMemory: gibQuantity(deviceMemoryBytes),
		LabelSizedAgainst: claim.SizedAgainst,
	}

	floor := fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("%dGi")) >= 0 && device.capacity["%s"].memory.compareTo(quantity("%dGi")) <= 0`,
		r.DeviceDriver, placement.CeilGiB(perDeviceBytes), r.DeviceDriver, placement.CeilGiB(deviceMemoryBytes))

	return &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claim.Name,
			Namespace: r.Namespace,
			Labels:    labels,
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{{
					Name: podClaimName,
					Exactly: &resourcev1.ExactDeviceRequest{
						DeviceClassName: r.DeviceClass,
						Count:           int64(claim.DeviceCount),
						AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
						Selectors: []resourcev1.DeviceSelector{{
							CEL: &resourcev1.CELDeviceSelector{Expression: floor},
						}},
					},
				}},
			},
		},
	}
}

// renderPod builds the worker pod: the operator's template for everything
// placement has no opinion about, the controller's decision for everything it
// does.
func (r *OpenRLWorkerReconciler) renderPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, podName, claimName string) (*corev1.Pod, error) {
	pod, err := r.loadTemplate(ctx, worker)
	if err != nil {
		return nil, err
	}
	if len(pod.Spec.Containers) == 0 {
		return nil, fmt.Errorf("pod template for role %s declares no containers", worker.Spec.Role)
	}

	pod.Name = podName
	pod.Namespace = r.Namespace
	applyOverlay(&pod.Spec.Containers[0], worker.Spec.Container)
	attachClaim(pod, claimName)
	attachTimeSliceGroup(pod, worker, claimName)

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels["app"] = "open-rl-" + string(worker.Spec.Role) + "-worker"
	pod.Labels[LabelClaim] = claimName
	// Label values cap at 63 characters and forbid dots; names do neither.
	// Labels carry sanitized identities, env vars carry the full ones.
	pod.Labels[LabelWorker] = sanitizeLabel(worker.Name)
	pod.Labels[LabelRole] = string(worker.Spec.Role)

	// Constraining the pod is what constrains where its claim can land;
	// whatever SKU or affinity the template pinned is dropped on purpose.
	// Two ORed terms, because a node selector cannot say "or unlabeled":
	// the documented default is that a node naming no role labels takes both.
	pod.Spec.NodeSelector = nil
	pod.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{
				{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: NodeLabelEnabled, Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
					{Key: nodeRoleLabel[worker.Spec.Role], Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
				}},
				{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: NodeLabelEnabled, Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
					{Key: NodeLabelTrainer, Operator: corev1.NodeSelectorOpDoesNotExist},
					{Key: NodeLabelSampler, Operator: corev1.NodeSelectorOpDoesNotExist},
				}},
			},
		},
	}}

	if err := controllerutil.SetControllerReference(worker, pod, r.Scheme()); err != nil {
		return nil, fmt.Errorf("set owner of pod %s: %w", podName, err)
	}
	return pod, nil
}

// loadTemplate reads the pod YAML the worker asked for, or the controller's
// role default.
func (r *OpenRLWorkerReconciler) loadTemplate(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker) (*corev1.Pod, error) {
	name, key := r.DefaultPodTemplates[worker.Spec.Role], "pod.yaml"
	if ref := worker.Spec.PodTemplate; ref != nil {
		name = ref.Name
		if ref.Key != "" {
			key = ref.Key
		}
	}
	if name == "" {
		return nil, fmt.Errorf("no pod template configured for role %s and none given in spec.podTemplate", worker.Spec.Role)
	}

	var cm corev1.ConfigMap
	if err := r.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: name}, &cm); err != nil {
		return nil, fmt.Errorf("read pod template ConfigMap %s: %w", name, err)
	}
	raw, ok := cm.Data[key]
	if !ok && len(cm.Data) == 1 {
		// A single-key ConfigMap is unambiguous; the deployed templates keep
		// the static worker manager's key names.
		for _, raw = range cm.Data {
			ok = true
		}
	}
	if !ok {
		return nil, fmt.Errorf("pod template ConfigMap %s has no key %q and does not have exactly one key", name, key)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal([]byte(raw), &pod); err != nil {
		return nil, fmt.Errorf("parse pod template %s/%s: %w", name, key, err)
	}
	return &pod, nil
}

// applyOverlay stamps the gateway's per-model decisions onto the container,
// so the controller never reads the gateway's metadata store.
func applyOverlay(container *corev1.Container, overlay *openrlv1alpha1.ContainerOverlay) {
	if overlay == nil {
		return
	}
	if overlay.Image != "" {
		container.Image = overlay.Image
	}
	if len(overlay.Command) > 0 {
		container.Command = overlay.Command
	}
	container.Args = append(container.Args, overlay.Args...)
	for _, env := range overlay.Env {
		setEnv(container, env)
	}
}

// setEnv merges one variable by name, overwriting whatever the template had.
func setEnv(container *corev1.Container, want corev1.EnvVar) {
	for i := range container.Env {
		if container.Env[i].Name == want.Name {
			container.Env[i] = want
			return
		}
	}
	container.Env = append(container.Env, want)
}

// attachClaim points the pod at a specific ResourceClaim, replacing whatever
// the template carried -- two GPU claims would pin the pod to the
// intersection of two allocations.
func attachClaim(pod *corev1.Pod, claimName string) {
	pod.Spec.ResourceClaims = []corev1.PodResourceClaim{{
		Name:              podClaimName,
		ResourceClaimName: &claimName,
	}}
	for i := range pod.Spec.Containers {
		pod.Spec.Containers[i].Resources.Claims = []corev1.ResourceClaim{{Name: podClaimName}}
	}
}

// attachTimeSliceGroup tells the node-local time-slicer which accelerator
// bundle this worker shares (group = the claim, so unrelated allocations
// never wait on each other) and which owner it is served under (turns rotate
// between owners; one worker resident at a time regardless).
func attachTimeSliceGroup(pod *corev1.Pod, worker *openrlv1alpha1.OpenRLWorker, claimName string) {
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	// The time-slice job id is the placement WorkerID: one identity formula,
	// so the booking key and the runtime key can never drift apart. Env vars
	// carry the exact values; the labels are sanitized copies for discovery.
	request := requestFrom(worker)
	jobID := request.WorkerID
	owner := request.OwnerKey()

	pod.Labels[timeSliceEnabledLabel] = "true"
	pod.Labels[timeSliceGroupLabel] = claimName
	pod.Labels[timeSliceOwnerLabel] = sanitizeLabel(owner)
	pod.Labels[timeSliceJobIDLabel] = sanitizeLabel(jobID)
	for i := range pod.Spec.Containers {
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceGroupEnv, Value: claimName})
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceOwnerEnv, Value: owner})
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceJobIDEnv, Value: jobID})
	}
}

// unschedulableMessage is the scheduler's reason a pod cannot be placed, if it
// gave one.
func unschedulableMessage(pod *corev1.Pod) string {
	if pod.Status.Phase != corev1.PodPending && pod.Status.Phase != "" {
		return ""
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type != corev1.PodScheduled || condition.Status != corev1.ConditionFalse {
			continue
		}
		detail := condition.Message
		if detail == "" {
			detail = condition.Reason
		}
		if detail != "" {
			return "Unschedulable: " + detail
		}
	}
	return ""
}
