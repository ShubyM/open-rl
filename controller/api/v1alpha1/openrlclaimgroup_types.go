package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Seat is one workload's booking on a shared claim.
type Seat struct {
	// Workload is the OpenRLWorkload holding this seat. One seat per
	// workload: booking is idempotent by name.
	Workload string `json:"workload"`

	// WorkloadUID pins the seat to one incarnation of that workload. A
	// deleted-and-recreated workload is a different identity; its stale
	// seat must not be mistaken for the new one's booking.
	WorkloadUID string `json:"workloadUID,omitempty"`

	// AssignmentID names this particular booking. The same value is written
	// to the workload's status and stamped into its pod, so the runtime can
	// refuse an obsolete pod whose booking has been superseded.
	AssignmentID string `json:"assignmentID,omitempty"`

	// Owner is the fairness unit the workload is served under, copied here
	// so admission can prefer less-contended groups without reading every
	// workload.
	Owner string `json:"owner,omitempty"`

	// HostRequest is the workload pod's memory request, copied here so
	// admission can sum what the node must satisfy without reading every
	// workload.
	HostRequest resource.Quantity `json:"hostRequest,omitempty"`
}

// OpenRLClaimGroupSpec is the seating chart for one managed ResourceClaim.
// Membership lives here and nowhere else; updates race through
// resourceVersion, so two controllers booking the last seat cannot both
// win -- one write lands, the other conflicts, re-reads, and re-checks.
type OpenRLClaimGroupSpec struct {
	// ClaimName is the ResourceClaim this group seats. The group shares the
	// claim's name; this field makes the pairing survive a rename audit.
	ClaimName string `json:"claimName"`

	// MaxSeats is the operator guardrail copied from the node label: an
	// operational ceiling, not a fairness guarantee. 0 means the ceiling is
	// not yet known -- the group was created before its claim allocated, and
	// the first booking that knows the node's policy stamps it.
	// +kubebuilder:validation:Minimum=0
	MaxSeats int32 `json:"maxSeats,omitempty"`

	// Seats is who holds the claim's membership. At most one seated
	// workload is resident on the devices at a time; the rest are
	// suspended. The list is the authority: worker statuses are
	// observational copies.
	// +listType=map
	// +listMapKey=workload
	Seats []Seat `json:"seats,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=ocg
// +kubebuilder:printcolumn:name="Claim",type=string,JSONPath=`.spec.claimName`
// +kubebuilder:printcolumn:name="Seats",type=string,JSONPath=`.spec.seats[*].workload`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// OpenRLClaimGroup records who sits on one shared ResourceClaim. It exists
// so that seat booking is arbitrated by etcd instead of by controller
// serialization: reconciles run concurrently, and a stale booking costs a
// conflict retry rather than a double-booked device.
type OpenRLClaimGroup struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OpenRLClaimGroupSpec `json:"spec"`
}

// +kubebuilder:object:root=true

// OpenRLClaimGroupList contains a list of OpenRLClaimGroup.
type OpenRLClaimGroupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []OpenRLClaimGroup `json:"items"`
}

func init() {
	SchemeBuilder.Register(&OpenRLClaimGroup{}, &OpenRLClaimGroupList{})
}
