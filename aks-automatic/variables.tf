variable "location" {
  description = "Location of the resource group."
  type        = string
  default     = "centralindia"
}

variable "project_prefix" {
  description = "Prefix of the resource group name that's combined with a random ID so name is unique in your Azure subscription."
  type        = string
  default     = "raydemo"
}

variable "resource_group_owner" {
  description = "The owner of the resource group."
  type        = string
}

variable "subscription_id" {
  description = "The Azure subscription ID."
  type        = string
}

variable "system_node_pool_vm_size" {
  description = "The size of the Virtual Machine."
  type        = string
  default     = "standard_D4lds_v5"
}

variable "system_node_pool_node_count" {
  description = "The initial quantity of nodes for the system node pool."
  type        = number
  default     = 1
}

variable "ray_node_pool_vm_size" {
  description = "The size of the Virtual Machine."
  type        = string
  default     = "Standard_NC40ads_H100_v5"
}

variable "username" {
  description = "The username for the Linux profile."
  type        = string
  default     = "azureuser"
}

variable "ksm_allowed_annotations" {
  description = "The allowed annotations for kube-state-metrics."
  type        = string
}

variable "ksm_allowed_labels" {
  description = "The allowed labels for kube-state-metrics."
  type        = string
}