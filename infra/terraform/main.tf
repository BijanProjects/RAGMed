# The RAGMed server on Oracle Cloud, kept inside the Always Free allowance:
# one Arm VM (VM.Standard.A1.Flex, 4 OCPUs, 24 GB), a 100 GB boot volume (200 GB is free),
# and a small network. No load balancer, no paid shapes.
#
# Credentials come from ~/.oci/config. Run through scripts/infra.sh, which fills in the
# tenancy and your current IP address for SSH.

terraform {
  required_version = ">= 1.9"
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 7.0"
    }
  }
}

provider "oci" {
  region              = var.region
  config_file_profile = "DEFAULT"
}

variable "region" {
  type    = string
  default = "ca-montreal-1"
}

variable "tenancy_ocid" {
  type        = string
  description = "From ~/.oci/config; the server goes in the root compartment."
}

variable "ssh_public_key" {
  type = string
}

variable "ssh_cidr" {
  type        = string
  description = "Only this address range may connect over SSH (your current IP, /32)."
}

locals {
  name = "ragmed"
  # The Always Free limit for Arm servers: 4 OCPUs and 24 GB in total.
  ocpus     = 4
  memory_gb = 24
  disk_gb   = 100
}

data "oci_identity_availability_domains" "all" {
  compartment_id = var.tenancy_ocid
}

data "oci_core_images" "ubuntu" {
  compartment_id           = var.tenancy_ocid
  operating_system         = "Canonical Ubuntu"
  operating_system_version = "24.04"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"
}

resource "oci_core_vcn" "main" {
  compartment_id = var.tenancy_ocid
  cidr_blocks    = ["10.0.0.0/16"]
  display_name   = local.name
  dns_label      = local.name
}

resource "oci_core_internet_gateway" "main" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = local.name
}

resource "oci_core_route_table" "public" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${local.name}-public"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.main.id
  }
}

resource "oci_core_security_list" "public" {
  compartment_id = var.tenancy_ocid
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${local.name}-public"

  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH only from your own address.
  ingress_security_rules {
    source   = var.ssh_cidr
    protocol = "6"
    tcp_options {
      min = 22
      max = 22
    }
  }

  # HTTPS (and HTTP, which Caddy redirects and uses for certificates).
  dynamic "ingress_security_rules" {
    for_each = [80, 443]
    content {
      source   = "0.0.0.0/0"
      protocol = "6"
      tcp_options {
        min = ingress_security_rules.value
        max = ingress_security_rules.value
      }
    }
  }

  ingress_security_rules {
    source   = "0.0.0.0/0"
    protocol = "17" # HTTP/3
    udp_options {
      min = 443
      max = 443
    }
  }

  # Path MTU discovery.
  ingress_security_rules {
    source   = "0.0.0.0/0"
    protocol = "1"
    icmp_options {
      type = 3
      code = 4
    }
  }
}

resource "oci_core_subnet" "public" {
  compartment_id    = var.tenancy_ocid
  vcn_id            = oci_core_vcn.main.id
  cidr_block        = "10.0.1.0/24"
  display_name      = "${local.name}-public"
  dns_label         = "public"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.public.id]
}

resource "oci_core_instance" "server" {
  compartment_id      = var.tenancy_ocid
  availability_domain = data.oci_identity_availability_domains.all.availability_domains[0].name
  display_name        = local.name
  shape               = "VM.Standard.A1.Flex"

  shape_config {
    ocpus         = local.ocpus
    memory_in_gbs = local.memory_gb
  }

  source_details {
    source_type             = "image"
    source_id               = data.oci_core_images.ubuntu.images[0].id
    boot_volume_size_in_gbs = local.disk_gb
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
    hostname_label   = local.name
  }

  metadata = {
    ssh_authorized_keys = var.ssh_public_key
    user_data           = base64encode(file("${path.module}/cloud-init.yaml"))
  }

  lifecycle {
    # A newer Ubuntu image or a changed setup script must never rebuild the running server.
    ignore_changes = [source_details[0].source_id, metadata]
  }
}

output "public_ip" {
  value = oci_core_instance.server.public_ip
}

output "shape" {
  value = "${oci_core_instance.server.shape} (${local.ocpus} OCPUs, ${local.memory_gb} GB, ${local.disk_gb} GB disk)"
}
