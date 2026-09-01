package v1alpha1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// Seat is one workload's booking on a shared claim.
type Seat struct {
	// Workload is the name of the Workload that holds this seat.
	// A workload holds at most one seat.
	Workload string `json:"workload"`

	// WorkloadUID is the Kubernetes UID of that Workload. It tells a
	// deleted-and-recreated workload apart from the one that booked this
	// seat, so the new one does not reuse the old seat.
	WorkloadUID types.UID `json:"workloadUID,omitempty"`

	// AssignmentID identifies this booking. The same value is written to
	// the workload status and its pod; a pod carrying an old value is
	// refused GPU access.
	AssignmentID string `json:"assignmentID,omitempty"`

	// OwnerID groups workloads that share one GPU turn; the API server sets
	// it (the job for FFT workers, the shared base model for LoRA workers).
	// Copied from the workload's spec.ownerId.
	OwnerID string `json:"ownerId,omitempty"`

	// HostRequest is the pod's host memory request. Copied here so
	// placement can add up a node's load without reading every workload.
	HostRequest resource.Quantity `json:"hostRequest,omitempty"`
}

// ClaimLedgerSpec is the seating chart for one managed ResourceClaim.
// Membership lives here and nowhere else; updates race through
// resourceVersion, so two controllers booking the last seat cannot both
// win -- one write lands, the other conflicts, re-reads, and re-checks.
type ClaimLedgerSpec struct {
	// ClaimName is the ResourceClaim this ledger seats. The ledger shares the
	// claim's name; this field makes the pairing survive a rename audit.
	ClaimName string `json:"claimName"`

	// Seats is who holds the claim's membership. At most one seated
	// workload is resident on the devices at a time; the rest are
	// suspended. The list is the authority: worker statuses are
	// observational copies.
	// +listType=map
	// +listMapKey=workload
	Seats []Seat `json:"seats,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=ocl
// +kubebuilder:printcolumn:name="Claim",type=string,JSONPath=`.spec.claimName`
// +kubebuilder:printcolumn:name="Seats",type=string,JSONPath=`.spec.seats[*].workload`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// ClaimLedger records who sits on one shared ResourceClaim. It exists
// so that seat booking is arbitrated by etcd instead of by controller
// serialization: reconciles run concurrently, and a stale booking costs a
// conflict retry rather than a double-booked device.
type ClaimLedger struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec ClaimLedgerSpec `json:"spec"`
}

// +kubebuilder:object:root=true

// ClaimLedgerList contains a list of ClaimLedger.
type ClaimLedgerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ClaimLedger `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ClaimLedger{}, &ClaimLedgerList{})
}
