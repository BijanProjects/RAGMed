import type { Metadata } from "next";

import { ClinicianView } from "@/components/clinician";

export const metadata: Metadata = { title: "Clinician view · RAGMed Clinic" };

export default function ClinicianPage() {
  return <ClinicianView />;
}
