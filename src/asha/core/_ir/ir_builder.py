# Copyright 2026 Ajay Rajan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import List, Dict, Any, Optional
from .intent_classifier import get_intent_classifier
from .prompt_ir import PromptIR, IntentType, EntityType, ConstraintType, PrivacyLevel


class IRBuilder:
    """
    IR Builder - Converts raw prompts into structured Prompt IR.

    Uses NLP patterns, rule-based extraction, and heuristics to parse
    user prompts into structured intermediate representations.
    """

    def __init__(self) -> None:
        """Initialize IR Builder with patterns and rules."""
        self.intent_patterns = self._init_intent_patterns()
        self.entity_patterns = self._init_entity_patterns()
        self.constraint_patterns = self._init_constraint_patterns()
        self.privacy_keywords = self._init_privacy_keywords()

    def _init_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize intent detection patterns."""
        return {
            IntentType.ANALYZE: [
                r"\banalyze\b",
                r"\bexamine\b",
                r"\binvestigate\b",
                r"\bstudy\b",
                r"\breview\b",
                r"\binspect\b",
                r"\bevaluate\b",
                r"\bassess\b",
            ],
            IntentType.GENERATE: [
                r"\bgenerate\b",
                r"\bcreate\b",
                r"\bproduce\b",
                r"\bwrite\b",
                r"\bmake\b",
                r"\bbuild\b",
                r"\bdevelop\b",
                r"\bcompose\b",
            ],
            IntentType.SUMMARIZE: [
                r"\bsummarize\b",
                r"\bsummarise\b",
                r"\bcondense\b",
                r"\bbrief\b",
                r"\bshorten\b",
                r"\babstract\b",
                r"\boutline\b",
                r"\brecap\b",
            ],
            IntentType.TRANSLATE: [
                r"\btranslate\b",
                r"\bconvert\b",
                r"\btransform\b",
                r"\badapt\b",
                r"\blocalize\b",
                r"\btranscribe\b",
            ],
            IntentType.CLASSIFY: [
                r"\bclassify\b",
                r"\bcategorize\b",
                r"\bcategorise\b",
                r"\bgroup\b",
                r"\bsort\b",
                r"\blabel\b",
                r"\btag\b",
                r"\borganize\b",
            ],
            IntentType.EXTRACT: [
                r"\bextract\b",
                r"\bpull\b",
                r"\bget\b",
                r"\bretrieve\b",
                r"\bobtain\b",
                r"\bfind\b",
                r"\blocate\b",
                r"\bidentify\b",
            ],
            IntentType.COMPARE: [
                r"\bcompare\b",
                r"\bcontrast\b",
                r"\bdiff\b",
                r"\bversus\b",
                r"\bvs\b",
                r"\bagainst\b",
                r"\bdifference\b",
            ],
            IntentType.EXPLAIN: [
                r"\bexplain\b",
                r"\bdescribe\b",
                r"\bdetail\b",
                r"\belaborate\b",
                r"\bclarify\b",
                r"\bdefine\b",
                r"\binterpret\b",
            ],
            IntentType.CREATE: [
                r"\bcreate\b",
                r"\bdesign\b",
                r"\bbuild\b",
                r"\bconstruct\b",
                r"\bmake\b",
                r"\bdevelop\b",
                r"\bform\b",
                r"\bcraft\b",
            ],
            IntentType.MODIFY: [
                r"\bmodify\b",
                r"\bchange\b",
                r"\badjust\b",
                r"\balter\b",
                r"\bedit\b",
                r"\bupdate\b",
                r"\brevise\b",
                r"\btweak\b",
            ],
            IntentType.VALIDATE: [
                r"\bvalidate\b",
                r"\bverify\b",
                r"\bcheck\b",
                r"\bconfirm\b",
                r"\btest\b",
                r"\bensure\b",
                r"\bguarantee\b",
            ],
            IntentType.SEARCH: [
                r"\bsearch\b",
                r"\bfind\b",
                r"\blook for\b",
                r"\blocate\b",
                r"\bquery\b",
                r"\bseek\b",
                r"\bhunt\b",
            ],
            IntentType.DEBUG: [
                r"\bdebug\b",
                r"\bfix\b",
                r"\btroubleshoot\b",
                r"\brepair\b",
                r"\bsolve\b",
                r"\bresolve\b",
                r"\bcorrect\b",
            ],
            IntentType.OPTIMIZE: [
                r"\boptimize\b",
                r"\bimprove\b",
                r"\benhance\b",
                r"\bboost\b",
                r"\bstreamline\b",
                r"\bfine-tune\b",
                r"\brefine\b",
            ],
        }

    def _init_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize entity detection patterns."""
        return {
            EntityType.DATASET: [
                r"\bdataset\b",
                r"\bdata\b",
                r"\bcsv\b",
                r"\bexcel\b",
                r"\btable\b",
                r"\bdatabase\b",
                r"\brecords\b",
                r"\bentries\b",
                r"\brows\b",
            ],
            EntityType.TEXT: [
                r"\btext\b",
                r"\bdocument\b",
                r"\barticle\b",
                r"\bparagraph\b",
                r"\bsentence\b",
                r"\bcontent\b",
                r"\bwriting\b",
                r"\bnarrative\b",
            ],
            EntityType.CODE: [
                r"\bcode\b",
                r"\bprogram\b",
                r"\bscript\b",
                r"\bfunction\b",
                r"\bclass\b",
                r"\bmethod\b",
                r"\balgorithm\b",
                r"\bsoftware\b",
            ],
            EntityType.DOCUMENT: [
                r"\bdocument\b",
                r"\bfile\b",
                r"\bpdf\b",
                r"\bword\b",
                r"\breport\b",
                r"\bpaper\b",
                r"\bmanual\b",
                r"\bguide\b",
            ],
            EntityType.IMAGE: [
                r"\bimage\b",
                r"\bpicture\b",
                r"\bphoto\b",
                r"\bgraphic\b",
                r"\bvisual\b",
                r"\bdiagram\b",
                r"\bchart\b",
                r"\bgraph\b",
            ],
            EntityType.DATA: [
                r"\bdata\b",
                r"\binformation\b",
                r"\binfo\b",
                r"\bstatistics\b",
                r"\bmetrics\b",
                r"\bnumbers\b",
                r"\bfigures\b",
            ],
            EntityType.MODEL: [
                r"\bmodel\b",
                r"\bml model\b",
                r"\bai model\b",
                r"\bneural\b",
                r"\bnetwork\b",
                r"\bclassifier\b",
                r"\bregressor\b",
            ],
            EntityType.API: [
                r"\bapi\b",
                r"\bservice\b",
                r"\bendpoint\b",
                r"\binterface\b",
                r"\brest\b",
                r"\bgraphql\b",
                r"\bwebhook\b",
            ],
            EntityType.SYSTEM: [
                r"\bsystem\b",
                r"\bapplication\b",
                r"\bapp\b",
                r"\bsoftware\b",
                r"\bplatform\b",
                r"\bframework\b",
                r"\btool\b",
            ],
            EntityType.USER: [
                r"\buser\b",
                r"\bcustomer\b",
                r"\bclient\b",
                r"\bperson\b",
                r"\bindividual\b",
                r"\bpeople\b",
                r"\baccount\b",
            ],
            EntityType.BUSINESS: [
                r"\bbusiness\b",
                r"\bcompany\b",
                r"\borganization\b",
                r"\bfirm\b",
                r"\bcorporation\b",
                r"\benterprise\b",
                r"\bstartup\b",
            ],
            EntityType.FINANCIAL: [
                r"\bfinancial\b",
                r"\bmoney\b",
                r"\bpayment\b",
                r"\btransaction\b",
                r"\bbilling\b",
                r"\binvoice\b",
                r"\bbudget\b",
                r"\bcost\b",
            ],
            EntityType.MEDICAL: [
                r"\bmedical\b",
                r"\bhealth\b",
                r"\bpatient\b",
                r"\bdiagnosis\b",
                r"\btreatment\b",
                r"\bclinical\b",
                r"\bmedicine\b",
                r"\bdoctor\b",
            ],
            EntityType.LEGAL: [
                r"\blegal\b",
                r"\blaw\b",
                r"\bcontract\b",
                r"\bagreement\b",
                r"\bregulation\b",
                r"\bcompliance\b",
                r"\bpolicy\b",
                r"\bterms\b",
            ],
        }

    def _init_constraint_patterns(self) -> Dict[ConstraintType, List[str]]:
        """Initialize constraint detection patterns."""
        return {
            ConstraintType.PRIVACY: [
                r"\bprivate\b",
                r"\bconfidential\b",
                r"\bsensitive\b",
                r"\bsecure\b",
                r"\bprotect\b",
                r"\bmask\b",
                r"\banonymize\b",
                r"\bencrypt\b",
            ],
            ConstraintType.ACCURACY: [
                r"\baccurate\b",
                r"\bprecise\b",
                r"\bexact\b",
                r"\bcorrect\b",
                r"\bperfect\b",
                r"\bflawless\b",
                r"\berror-free\b",
            ],
            ConstraintType.SPEED: [
                r"\bfast\b",
                r"\bquick\b",
                r"\brapid\b",
                r"\bspeedy\b",
                r"\bimmediate\b",
                r"\binstant\b",
                r"\burst\b",
            ],
            ConstraintType.COST: [
                r"\bcheap\b",
                r"\baffordable\b",
                r"\blow-cost\b",
                r"\beconomical\b",
                r"\bbudget\b",
                r"\bfree\b",
                r"\bcost-effective\b",
            ],
            ConstraintType.FORMAT: [
                r"\bformat\b",
                r"\bstyle\b",
                r"\btemplate\b",
                r"\bstructure\b",
                r"\blayout\b",
                r"\bschema\b",
                r"\bpattern\b",
            ],
            ConstraintType.LENGTH: [
                r"\bshort\b",
                r"\bbrief\b",
                r"\bconcise\b",
                r"\bterse\b",
                r"\blong\b",
                r"\bdetailed\b",
                r"\bcomprehensive\b",
                r"\bthorough\b",
            ],
            ConstraintType.STYLE: [
                r"\bformal\b",
                r"\bcasual\b",
                r"\bprofessional\b",
                r"\bfriendly\b",
                r"\btechnical\b",
                r"\bsimple\b",
                r"\bclear\b",
                r"\bdirect\b",
            ],
            ConstraintType.LANGUAGE: [
                r"\benglish\b",
                r"\bspanish\b",
                r"\bfrench\b",
                r"\bgerman\b",
                r"\bchinese\b",
                r"\bjapanese\b",
                r"\brussian\b",
                r"\barabic\b",
            ],
            ConstraintType.SECURITY: [
                r"\bsecure\b",
                r"\bsafe\b",
                r"\bprotected\b",
                r"\bencrypted\b",
                r"\bauthenticated\b",
                r"\bauthorized\b",
                r"\bverified\b",
            ],
            ConstraintType.COMPLIANCE: [
                r"\bgdpr\b",
                r"\bhipaa\b",
                r"\bsoc2\b",
                r"\bcompliant\b",
                r"\bregulation\b",
                r"\bpolicy\b",
                r"\bstandard\b",
            ],
        }

    def _init_privacy_keywords(self) -> Dict[PrivacyLevel, List[str]]:
        """Initialize privacy level detection keywords."""
        return {
            PrivacyLevel.PUBLIC: [
                "public",
                "open",
                "shared",
                "published",
                "accessible",
            ],
            PrivacyLevel.INTERNAL: [
                "internal",
                "company",
                "organization",
                "internal use",
            ],
            PrivacyLevel.CONFIDENTIAL: [
                "confidential",
                "proprietary",
                "secret",
                "restricted",
            ],
            PrivacyLevel.RESTRICTED: [
                "restricted",
                "classified",
                "top secret",
                "highly sensitive",
            ],
            PrivacyLevel.MASKED: [
                "mask",
                "anonymize",
                "redact",
                "hide sensitive",
                "remove pii",
            ],
        }

    def _get_ast_analysis(self, prompt: str) -> Dict[str, Any]:
        """
        Get AST analysis using the parser module.

        Args:
            prompt: Normalized prompt text

        Returns:
            AST analysis results
        """
        try:
            from ..parser.prompt_ast import PromptParser

            parser = PromptParser()
            ast = parser.parse(prompt)
            if hasattr(ast, "intent"):
                primary_object = (
                    ast.objects[0] if getattr(ast, "objects", None) else None
                )
                return {
                    "intent": (
                        ast.intent.value if hasattr(ast.intent, "value") else ast.intent
                    ),
                    "object": (
                        primary_object.value
                        if primary_object and hasattr(primary_object, "value")
                        else primary_object
                    ),
                    "entities": getattr(ast, "entities", []),
                    "keywords": getattr(ast, "keywords", []),
                }
            if isinstance(ast, dict):
                return ast
            return {"intent": None, "object": None}
        except ImportError:
            # Fallback if parser not available
            return {"intent": None, "object": None}
        except Exception as e:
            # Fail-safe on parser error
            return {"intent": None, "object": None, "error": str(e)}

    def parse(self, raw_prompt: str) -> PromptIR:
        """
        Parse raw prompt into structured Prompt IR.

        Args:
            raw_prompt: Raw user prompt text

        Returns:
            Structured Prompt IR representation
        """
        # Normalize prompt
        normalized_prompt = self._normalize_prompt(raw_prompt)

        # Enhanced parsing with AST analysis
        ast_analysis = self._get_ast_analysis(normalized_prompt)

        # Extract components
        intent = self._extract_intent(normalized_prompt, ast_analysis)
        entity = self._extract_entity(normalized_prompt, ast_analysis)
        constraints = self._extract_constraints(normalized_prompt)
        privacy = self._extract_privacy_level(normalized_prompt)
        extracted_entities = self._extract_named_entities(normalized_prompt)
        parameters = self._extract_parameters(normalized_prompt)

        # Calculate metrics
        complexity_score = self._calculate_complexity(
            normalized_prompt, intent, constraints
        )
        token_estimate = self._estimate_tokens(normalized_prompt)

        # Determine urgency
        urgency = self._determine_urgency(normalized_prompt)

        # Extract context
        context = self._extract_context(normalized_prompt)

        return PromptIR(
            intent=intent,
            entity=entity,
            constraints=constraints,
            privacy=privacy,
            original_prompt=raw_prompt,
            extracted_entities=extracted_entities,
            parameters=parameters,
            context=context,
            urgency=urgency,
            complexity_score=complexity_score,
            token_estimate=token_estimate,
        )

    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize prompt for processing."""
        # Convert to lowercase and strip
        normalized = prompt.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove common conversational fillers
        fillers = [
            r"\bhey\b",
            r"\bhi\b",
            r"\bhello\b",
            r"\bplease\b",
            r"\bcan you\b",
            r"\bcould you\b",
            r"\bwould you\b",
            r"\bhelp me\b",
            r"\bi need\b",
            r"\bi want\b",
        ]

        for filler in fillers:
            normalized = re.sub(filler, "", normalized)

        return normalized.strip()

    def _extract_intent(
        self, prompt: str, ast_analysis: Optional[Dict[str, Any]] = None
    ) -> IntentType:
        """Extract primary intent; abstain when confidence is below safe_max."""
        if not prompt.strip():
            return IntentType.ABSTAIN

        ast_hint = None
        if ast_analysis and ast_analysis.get("intent"):
            ast_hint = str(ast_analysis["intent"])

        prediction = get_intent_classifier().predict(prompt, ast_hint=ast_hint)
        if prediction.abstained:
            return IntentType.ABSTAIN
        return prediction.intent

    def _extract_entity(
        self, prompt: str, ast_analysis: Optional[Dict[str, Any]] = None
    ) -> EntityType:
        """Extract primary entity from prompt with AST analysis enhancement."""
        entity_scores: Dict[EntityType, int] = {}

        # Enhanced entity extraction with AST analysis
        if ast_analysis and ast_analysis.get("object"):
            ast_object = ast_analysis["object"]
            if ast_object == "dataset":
                entity_scores[EntityType.DATA] = (
                    entity_scores.get(EntityType.DATA, 0) + 3
                )
            elif ast_object == "text":
                entity_scores[EntityType.TEXT] = (
                    entity_scores.get(EntityType.TEXT, 0) + 3
                )
            elif ast_object == "image":
                entity_scores[EntityType.IMAGE] = (
                    entity_scores.get(EntityType.IMAGE, 0) + 3
                )

        # Traditional pattern matching
        for entity, patterns in self.entity_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt, re.IGNORECASE))
                score += matches
            entity_scores[entity] = entity_scores.get(entity, 0) + score

        # Return entity with highest score, default to DATA
        if entity_scores:
            best_entity = max(entity_scores, key=lambda k: entity_scores[k])
            return best_entity if entity_scores[best_entity] > 0 else EntityType.DATA

        return EntityType.DATA

    def _extract_constraints(self, prompt: str) -> List[ConstraintType]:
        """Extract constraints from prompt."""
        found_constraints = []

        for constraint, patterns in self.constraint_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    found_constraints.append(constraint)
                    break

        return found_constraints

    def _extract_privacy_level(self, prompt: str) -> PrivacyLevel:
        """Extract privacy level from prompt."""
        for privacy_level, keywords in self.privacy_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    return privacy_level

        # Default to INTERNAL if no privacy keywords found
        return PrivacyLevel.INTERNAL

    def _extract_named_entities(self, prompt: str) -> List[str]:
        """Extract specific named entities (simple regex-based)."""
        entities = []

        # Extract email addresses
        emails = re.findall(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", prompt
        )
        entities.extend(emails)

        # Extract phone numbers
        phones = re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", prompt)
        entities.extend(phones)

        # Extract URLs
        urls = re.findall(r"https?://[^\s]+", prompt)
        entities.extend(urls)

        # Extract file paths
        file_paths = re.findall(r"\b[A-Za-z]:\\[^\\s]+\b|\b/[^\s]+\b", prompt)
        entities.extend(file_paths)

        return entities

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract parameters and key-value pairs."""
        parameters = {}

        # Extract numbers
        numbers = re.findall(r"\b\d+\.?\d*\b", prompt)
        if numbers:
            parameters["numbers"] = [float(n) if "." in n else int(n) for n in numbers]

        # Extract quotes (could be specific values)
        quotes = re.findall(r'"([^"]*)"', prompt)
        if quotes:
            parameters["quoted_values"] = quotes

        # Extract boolean indicators
        if any(word in prompt for word in ["true", "yes", "enable", "on"]):
            parameters["boolean_indicators"] = [True]
        elif any(word in prompt for word in ["false", "no", "disable", "off"]):
            parameters["boolean_indicators"] = [False]

        return parameters

    def _calculate_complexity(
        self, prompt: str, intent: IntentType, constraints: List[ConstraintType]
    ) -> float:
        """Calculate complexity score (0.0 to 1.0)."""
        complexity = 0.0

        # Base complexity from prompt length
        length_score = min(len(prompt.split()) / 50, 1.0)  # Normalize to 0-1
        complexity += length_score * 0.3

        # Intent complexity
        intent_complexity = {
            IntentType.ANALYZE: 0.5,
            IntentType.GENERATE: 0.6,
            IntentType.OPTIMIZE: 0.8,
            IntentType.DEBUG: 0.9,
            IntentType.CREATE: 0.7,
            IntentType.MODIFY: 0.6,
            IntentType.VALIDATE: 0.7,
        }
        complexity += intent_complexity.get(intent, 0.5) * 0.4

        # Constraint complexity
        constraint_weight = len(constraints) * 0.1
        complexity += min(constraint_weight, 0.3)

        return min(complexity, 1.0)

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimation: ~1.3 tokens per word
        word_count = len(prompt.split())
        return int(word_count * 1.3)

    def _determine_urgency(self, prompt: str) -> Optional[str]:
        """Determine urgency level."""
        urgent_keywords = ["urgent", "asap", "immediately", "quickly", "fast", "now"]
        high_keywords = ["important", "priority", "critical"]

        if any(keyword in prompt for keyword in urgent_keywords):
            return "high"
        elif any(keyword in prompt for keyword in high_keywords):
            return "medium"

        return None

    def _extract_context(self, prompt: str) -> Optional[str]:
        """Extract contextual information."""
        # Look for context indicators
        context_patterns = [
            r"(?:in|for|about|regarding)\s+([^,.!?]+)",
            r"(?:because|since|due to)\s+([^,.!?]+)",
            r"(?:context|background|info)\s*:?\s*([^,.!?]+)",
        ]

        for pattern in context_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                return str(matches[0]).strip()

        return None
