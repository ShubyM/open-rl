package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// WorkerRole is which half of the training loop a worker runs. It selects
// which node pools may host the worker, and nothing else.
// +kubebuilder:validation:Enum=trainer;sampler
type WorkerRole string

const (
	RoleTrainer WorkerRole = "trainer"
	RoleSampler WorkerRole = "sampler"
)

// Phase is a coarse summary of where a worker is in scheduling.
// +kubebuilder:validation:Enum=Pending;Placing;Running;Failed
type Phase string

const (
	// PhasePending means no claim has been assigned, usually for want of capacity.
	PhasePending Phase = "Pending"
	// PhasePlacing means a claim and pod exist but the allocation is not yet observed.
	PhasePlacing Phase = "Placing"
	// PhaseRunning means the pod is running on an observed allocation.
	PhaseRunning Phase = "Running"
	// PhaseFailed means the request cannot be satisfied as written.
	PhaseFailed Phase = "Failed"
)

// Condition types set on an OpenRLWorker.
const (
	// ConditionPlaced reports whether the worker holds a claim and a pod.
	ConditionPlaced = "Placed"
)

// PodTemplateRef names the ConfigMap holding the pod spec this worker is
// rendered from. The template carries what scheduling has no opinion about;
// the controller overwrites nodeSelector and resourceClaims, because those
// are the decision.
type PodTemplateRef struct {
	// Name of the ConfigMap in the controller's namespace.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Key within the ConfigMap holding the pod YAML.
	// +kubebuilder:default=pod.yaml
	Key string `json:"key,omitempty"`
}

// ContainerOverlay is what the caller stamps onto the template's first
// container: everything the pod needs to know about the model arrives here.
type ContainerOverlay struct {
	// Image replaces the template's image when set.
	Image string `json:"image,omitempty"`

	// Command replaces the template's command when non-empty.
	Command []string `json:"command,omitempty"`

	// Args are appended to the template's args.
	Args []string `json:"args,omitempty"`

	// Env entries are merged by name, overwriting the template's values.
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// OpenRLWorkerSpec is one worker process's scheduling request. Memory arrives
// already estimated; the controller decides how many devices to spread it
// over and which claim to put it on, and never re-estimates.
type OpenRLWorkerSpec struct {
	// Role selects which node pools may host this worker, via the
	// openrl.io/trainer and openrl.io/sampler labels. It does not partition
	// claims: on a node that accepts both, the two roles share by turns.
	Role WorkerRole `json:"role"`

	// ModelID names the model this worker serves: configuration for the
	// worker process and the humans reading kubectl get, never identity.
	// The worker's identity is metadata.name.
	// +kubebuilder:validation:MinLength=1
	ModelID string `json:"modelId"`

	// OwnerID is the unit of fairness: the runtime serves owners round-robin,
	// so an owner never gets extra turns for having more processes, requests,
	// or adapters. Opaque to the controller, never read by placement -- one
	// worker is resident at a time whatever the owners. A worker naming no
	// owner is an owner of one.
	OwnerID string `json:"ownerId,omitempty"`

	// Memory is the estimator's figure: total peak accelerator memory, an
	// aggregate across however many devices it takes. Never re-estimated.
	Memory resource.Quantity `json:"memory"`

	// EstimatorVersion records which estimator produced Memory, so a placement
	// can be inspected and reproduced later.
	EstimatorVersion string `json:"estimatorVersion,omitempty"`

	// PodTemplate names the ConfigMap the worker pod is rendered from. When
	// unset the controller falls back to its role-default template.
	PodTemplate *PodTemplateRef `json:"podTemplate,omitempty"`

	// Container is what the caller stamps onto the rendered container.
	Container *ContainerOverlay `json:"container,omitempty"`
}

// OpenRLWorkerStatus is what the controller decided and what came of it.
type OpenRLWorkerStatus struct {
	// Phase is a coarse summary; Conditions carry the detail.
	Phase Phase `json:"phase,omitempty"`

	// DeviceCount is how many accelerators the controller decided this worker
	// spans. A decision, not a request: derived from Memory and the capacity
	// the pools registered.
	DeviceCount int32 `json:"deviceCount,omitempty"`

	// MemoryPerDevice is what each of those devices must provide, i.e. Memory
	// divided by DeviceCount and rounded up.
	MemoryPerDevice string `json:"memoryPerDevice,omitempty"`

	// HostMemoryWhenParked is the host RAM this worker occupies while
	// suspended: its full accelerator footprint, parked in its own host
	// address space. This is what actually bounds how many workers a node
	// can carry.
	HostMemoryWhenParked string `json:"hostMemoryWhenParked,omitempty"`

	// EstimatorVersion echoes which estimator produced the memory figure this
	// placement was decided against, so the decision can be reproduced.
	EstimatorVersion string `json:"estimatorVersion,omitempty"`

	// ClaimName is the ResourceClaim this worker was assigned to.
	ClaimName string `json:"claimName,omitempty"`

	// PodName is the worker pod, once created.
	PodName string `json:"podName,omitempty"`

	// NodeName is set only once the claim's allocation is observed. Until then
	// the node is genuinely unknown.
	NodeName string `json:"nodeName,omitempty"`

	// Reason is the most recent human-readable explanation of Phase.
	Reason string `json:"reason,omitempty"`

	// ObservedGeneration is the spec generation this status was computed from.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions holds the Placed condition and its history.
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=orw
// +kubebuilder:printcolumn:name="Role",type=string,JSONPath=`.spec.role`
// +kubebuilder:printcolumn:name="Owner",type=string,JSONPath=`.spec.ownerId`
// +kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.status.deviceCount`
// +kubebuilder:printcolumn:name="MemEach",type=string,JSONPath=`.status.memoryPerDevice`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Claim",type=string,JSONPath=`.status.claimName`
// +kubebuilder:printcolumn:name="Node",type=string,JSONPath=`.status.nodeName`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// OpenRLWorker is the scheduling request for a single worker process. The
// caller creates one per worker it wants; the controller turns it into a
// ResourceClaim (matched or created) and a pod, and records what it picked in
// status -- so `kubectl get openrlworkers` shows what was asked for, where it
// went, and why a pending worker is still pending.
type OpenRLWorker struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// The spec is immutable: every field either places the worker or renders
	// its pod, and V1 does not re-place or re-render a live worker. Change by
	// deleting and recreating.
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="OpenRLWorker spec is immutable; delete and recreate the worker"
	Spec   OpenRLWorkerSpec   `json:"spec"`
	Status OpenRLWorkerStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// OpenRLWorkerList is a list of OpenRLWorker.
type OpenRLWorkerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []OpenRLWorker `json:"items"`
}

func init() {
	SchemeBuilder.Register(&OpenRLWorker{}, &OpenRLWorkerList{})
}
