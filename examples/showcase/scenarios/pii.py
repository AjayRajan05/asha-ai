# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PII test scenarios for ASHA Control Center."""

from typing import Dict

PII_SCENARIOS: Dict[str, str] = {
    "Customer Support Ticket": (
        "Customer Johnathan Miller (SSN: 123-45-6789) called from +1-555-019-2834 regarding "
        "order #94821. His email is j.miller@cybercorp.io and billing address is "
        "742 Evergreen Terrace, Springfield, OR. Please process his refund of $450 to "
        "Visa ending in 4111-2222-3333-4444."
    ),
    "Developer Debug Log & API Key": (
        "Encountered 500 error connecting to upstream service. Request initiated by developer "
        "Sarah Connor (sconnor@skynet-defense.mil). "
        "Authorization header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doNotLeakThisToken "
        "API Key used: sk-live-938210984091283091284091283"
    ),
    "Medical & Personal Record": (
        "Patient Jane Doe (DOB: 1984-07-12, SSN: 987-65-4321) visited Dr. Gregory House at "
        "100 Medical Center Way, Suite 400. Direct contact: jdoe84@medmail.org or +1 (415) 555-9988. "
        "Prescription renewal requested."
    ),
    "Clean Benign Text (No PII)": (
        "Please provide a Python function to compute the Fibonacci sequence using dynamic "
        "programming with memoization, including type hints and docstrings."
    ),
}
