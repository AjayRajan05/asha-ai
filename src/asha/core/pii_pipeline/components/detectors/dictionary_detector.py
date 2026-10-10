"""
Dictionary-based PII Detector - Dictionary and keyword-based detection
"""

import re
from typing import List, Dict, Set
from ...stages.base_stage import PIIEntity


class DictionaryDetector:
    """
    Dictionary-based PII detector using keyword matching and dictionaries.

    This detector uses pre-compiled dictionaries and keyword matching
    to identify PII that might not match exact patterns.
    """

    def __init__(self) -> None:
        """Initialize dictionary detector with PII dictionaries."""
        self.dictionaries = self._load_dictionaries()
        self.context_keywords = self._load_context_keywords()
        self.confidence_weights = self._get_confidence_weights()

    def _load_dictionaries(self) -> Dict[str, Set[str]]:
        """Load PII dictionaries."""
        dictionaries = {
            "first_names": {
                "john",
                "mary",
                "james",
                "patricia",
                "robert",
                "jennifer",
                "michael",
                "linda",
                "william",
                "elizabeth",
                "david",
                "barbara",
                "richard",
                "susan",
                "joseph",
                "jessica",
                "thomas",
                "sarah",
                "charles",
                "karen",
                "christopher",
                "nancy",
                "daniel",
                "lisa",
                "matthew",
                "betty",
                "anthony",
                "helen",
                "mark",
                "sandra",
                "donald",
                "donna",
                "steven",
                "carol",
                "paul",
                "ruth",
                "andrew",
                "sharon",
                "joshua",
                "michelle",
                "kevin",
                "laura",
                "brian",
                "sarah",
                "george",
                "kimberly",
                "edward",
                "deborah",
                "ronald",
                "dorothy",
                "timothy",
                "lisa",
                "jason",
                "nancy",
                "jeffrey",
                "karen",
            },
            "last_names": {
                "smith",
                "johnson",
                "williams",
                "brown",
                "jones",
                "garcia",
                "miller",
                "davis",
                "rodriguez",
                "martinez",
                "hernandez",
                "lopez",
                "gonzalez",
                "wilson",
                "anderson",
                "thomas",
                "taylor",
                "moore",
                "jackson",
                "martin",
                "lee",
                "perez",
                "thompson",
                "white",
                "harris",
                "sanchez",
                "clark",
                "ramirez",
                "lewis",
                "robinson",
                "walker",
                "young",
                "allen",
                "king",
                "wright",
                "scott",
                "torres",
                "nguyen",
                "hill",
                "flores",
                "green",
                "adams",
                "baker",
                "gomez",
                "nelson",
                "carter",
                "mitchell",
            },
            "cities": {
                "new york",
                "los angeles",
                "chicago",
                "houston",
                "phoenix",
                "philadelphia",
                "san antonio",
                "san diego",
                "dallas",
                "san jose",
                "austin",
                "jacksonville",
                "fort worth",
                "columbus",
                "charlotte",
                "san francisco",
                "indianapolis",
                "seattle",
                "denver",
                "washington",
                "boston",
                "el paso",
                "nashville",
                "detroit",
                "portland",
                "memphis",
                "oklahoma city",
                "las vegas",
                "louisville",
            },
            "states": {
                "alabama",
                "alaska",
                "arizona",
                "arkansas",
                "california",
                "colorado",
                "connecticut",
                "delaware",
                "florida",
                "georgia",
                "hawaii",
                "idaho",
                "illinois",
                "indiana",
                "iowa",
                "kansas",
                "kentucky",
                "louisiana",
                "maine",
                "maryland",
                "massachusetts",
                "michigan",
                "minnesota",
                "mississippi",
                "missouri",
                "montana",
                "nebraska",
                "nevada",
                "new hampshire",
                "new jersey",
                "new mexico",
                "new york",
                "north carolina",
                "north dakota",
                "ohio",
                "oklahoma",
                "oregon",
                "pennsylvania",
                "rhode island",
                "south carolina",
                "south dakota",
                "tennessee",
                "texas",
                "utah",
                "vermont",
                "virginia",
                "washington",
                "west virginia",
                "wisconsin",
                "wyoming",
            },
            "countries": {
                "united states",
                "canada",
                "mexico",
                "united kingdom",
                "france",
                "germany",
                "italy",
                "spain",
                "japan",
                "china",
                "india",
                "australia",
                "brazil",
                "russia",
                "south korea",
                "netherlands",
                "switzerland",
                "sweden",
                "norway",
                "denmark",
                "finland",
                "belgium",
                "austria",
                "poland",
                "czech republic",
                "hungary",
                "romania",
                "bulgaria",
                "greece",
                "portugal",
                "ireland",
            },
            "companies": {
                "microsoft",
                "apple",
                "google",
                "amazon",
                "facebook",
                "tesla",
                "netflix",
                "twitter",
                "linkedin",
                "instagram",
                "youtube",
                "tiktok",
                "snapchat",
                "uber",
                "lyft",
                "airbnb",
                "booking.com",
                "expedia",
                "tripadvisor",
                "walmart",
                "target",
                "costco",
                "home depot",
                "lowe's",
                "best buy",
                "starbucks",
                "mcdonald's",
                "burger king",
                "subway",
                "kfc",
                "pizza hut",
            },
            "job_titles": {
                "software engineer",
                "data scientist",
                "product manager",
                "project manager",
                "marketing manager",
                "sales manager",
                "financial analyst",
                "business analyst",
                "chief executive officer",
                "chief technology officer",
                "chief financial officer",
                "vice president",
                "director",
                "manager",
                "supervisor",
                "team lead",
                "senior engineer",
                "junior engineer",
                "lead developer",
                "full stack developer",
                "front end developer",
                "back end developer",
                "mobile developer",
                "devops engineer",
            },
        }

        return dictionaries

    def _load_context_keywords(self) -> Dict[str, List[str]]:
        """Load context keywords for PII detection."""
        return {
            "name": [
                "my name is",
                "i am",
                "call me",
                "known as",
                "my name's",
                "i'm",
                "contact",
                "reach me",
                "get in touch",
                "my name",
            ],
            "email": [
                "my email",
                "email me",
                "contact me at",
                "reach me at",
                "my address",
                "email address",
                "e-mail",
                "mail me",
                "send to",
            ],
            "phone": [
                "my phone",
                "call me",
                "my number",
                "contact me",
                "reach me",
                "phone number",
                "telephone",
                "mobile",
                "cell phone",
                "my mobile",
            ],
            "address": [
                "my address",
                "live at",
                "located at",
                "my home",
                "my residence",
                "address is",
                "living at",
                "staying at",
                "based in",
            ],
            "ssn": [
                "social security",
                "ssn",
                "my ssn",
                "social security number",
                "tax id",
                "taxpayer id",
                "identification number",
            ],
            "credit_card": [
                "credit card",
                "card number",
                "payment method",
                "my card",
                "debit card",
                "visa",
                "mastercard",
                "american express",
            ],
        }

    def _get_confidence_weights(self) -> Dict[str, float]:
        """Get confidence weights for different PII types."""
        return {
            "name": 0.70,
            "email": 0.80,
            "phone": 0.75,
            "address": 0.65,
            "ssn": 0.85,
            "credit_card": 0.80,
            "location": 0.60,
            "organization": 0.55,
            "job_title": 0.50,
        }

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities using dictionary matching.

        Args:
            text: Input text to analyze

        Returns:
            List of detected PII entities
        """
        entities = []
        text_lower = text.lower()

        # Detect names using dictionary matching
        name_entities = self._detect_names(text, text_lower)
        entities.extend(name_entities)

        # Detect locations using dictionary matching
        location_entities = self._detect_locations(text, text_lower)
        entities.extend(location_entities)

        # Detect organizations using dictionary matching
        org_entities = self._detect_organizations(text, text_lower)
        entities.extend(org_entities)

        # Detect PII using context keywords
        context_entities = self._detect_context_pii(text, text_lower)
        entities.extend(context_entities)

        return entities

    def _detect_names(self, text: str, text_lower: str) -> List[PIIEntity]:
        """Detect names using dictionary matching."""
        entities = []
        first_names = self.dictionaries["first_names"]
        last_names = self.dictionaries["last_names"]

        # Split text into words
        words = text.split()
        words_lower = text_lower.split()

        # Look for first name + last name combinations
        for i in range(len(words) - 1):
            word1_lower = words_lower[i].strip(".,!?;:")
            word2_lower = words_lower[i + 1].strip(".,!?;:")

            if word1_lower in first_names and word2_lower in last_names:
                # Find the original text position
                name_text = f"{words[i]} {words[i + 1]}"
                pattern = re.escape(name_text)
                match = re.search(pattern, text)

                if match:
                    # Check if it's in a name context
                    context = self._get_context(text, match.start(), match.end())
                    if self._is_name_context(context):
                        entity = PIIEntity(
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            pii_type="name",
                            confidence=0.75,  # Higher confidence for dictionary matches
                            context=context,
                            metadata={
                                "detector": "dictionary",
                                "first_name": word1_lower,
                                "last_name": word2_lower,
                                "validation_method": "dictionary_match",
                            },
                        )
                        entities.append(entity)

        return entities

    def _detect_locations(self, text: str, text_lower: str) -> List[PIIEntity]:
        """Detect locations using dictionary matching."""
        entities = []
        cities = self.dictionaries["cities"]
        states = self.dictionaries["states"]
        countries = self.dictionaries["countries"]

        # Combine all location dictionaries
        all_locations = cities | states | countries

        for location in sorted(
            all_locations, key=len, reverse=True
        ):  # Check longer matches first
            if location in text_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = text_lower.find(location, start)
                    if pos == -1:
                        break

                    # Check if it's a location context
                    context = self._get_context(text, pos, pos + len(location))
                    if self._is_location_context(context):
                        # Get the actual text from original text (preserving case)
                        actual_text = text[pos : pos + len(location)]

                        entity = PIIEntity(
                            text=actual_text,
                            start=pos,
                            end=pos + len(location),
                            pii_type="location",
                            confidence=0.65,
                            context=context,
                            metadata={
                                "detector": "dictionary",
                                "location_type": self._classify_location(location),
                                "validation_method": "dictionary_match",
                            },
                        )
                        entities.append(entity)

                    start = pos + 1

        return entities

    def _detect_organizations(self, text: str, text_lower: str) -> List[PIIEntity]:
        """Detect organizations using dictionary matching."""
        entities = []
        companies = self.dictionaries["companies"]
        job_titles = self.dictionaries["job_titles"]

        # Detect companies
        for company in sorted(companies, key=len, reverse=True):
            if company in text_lower:
                start = 0
                while True:
                    pos = text_lower.find(company, start)
                    if pos == -1:
                        break

                    context = self._get_context(text, pos, pos + len(company))
                    if self._is_organization_context(context):
                        actual_text = text[pos : pos + len(company)]

                        entity = PIIEntity(
                            text=actual_text,
                            start=pos,
                            end=pos + len(company),
                            pii_type="organization",
                            confidence=0.60,
                            context=context,
                            metadata={
                                "detector": "dictionary",
                                "org_type": "company",
                                "validation_method": "dictionary_match",
                            },
                        )
                        entities.append(entity)

                    start = pos + 1

        # Detect job titles
        for job_title in sorted(job_titles, key=len, reverse=True):
            if job_title in text_lower:
                start = 0
                while True:
                    pos = text_lower.find(job_title, start)
                    if pos == -1:
                        break

                    context = self._get_context(text, pos, pos + len(job_title))
                    if self._is_job_title_context(context):
                        actual_text = text[pos : pos + len(job_title)]

                        entity = PIIEntity(
                            text=actual_text,
                            start=pos,
                            end=pos + len(job_title),
                            pii_type="job_title",
                            confidence=0.55,
                            context=context,
                            metadata={
                                "detector": "dictionary",
                                "validation_method": "dictionary_match",
                            },
                        )
                        entities.append(entity)

                    start = pos + 1

        return entities

    def _detect_context_pii(self, text: str, text_lower: str) -> List[PIIEntity]:
        """Detect PII using context keywords."""
        entities = []

        for pii_type, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Look for PII patterns near the keyword
                    keyword_pos = text_lower.find(keyword)
                    if keyword_pos != -1:
                        # Search in a window around the keyword
                        start = max(0, keyword_pos - 50)
                        end = min(len(text), keyword_pos + len(keyword) + 50)
                        window = text[start:end]

                        # Try to find specific patterns in this window
                        if pii_type == "email":
                            email_match = re.search(
                                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                                window,
                            )
                            if email_match:
                                entity = PIIEntity(
                                    text=email_match.group(),
                                    start=start + email_match.start(),
                                    end=start + email_match.end(),
                                    pii_type="email",
                                    confidence=0.85,  # High confidence with context
                                    context=window,
                                    metadata={
                                        "detector": "dictionary",
                                        "context_keyword": keyword,
                                        "validation_method": "context_match",
                                    },
                                )
                                entities.append(entity)

                        elif pii_type == "phone":
                            phone_match = re.search(
                                r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", window
                            )
                            if phone_match:
                                entity = PIIEntity(
                                    text=phone_match.group(),
                                    start=start + phone_match.start(),
                                    end=start + phone_match.end(),
                                    pii_type="phone",
                                    confidence=0.80,
                                    context=window,
                                    metadata={
                                        "detector": "dictionary",
                                        "context_keyword": keyword,
                                        "validation_method": "context_match",
                                    },
                                )
                                entities.append(entity)

        return entities

    def _is_name_context(self, context: str) -> bool:
        """Check if context indicates a name."""
        context_lower = context.lower()
        name_indicators = ["name", "called", "known as", "my name", "i am", "contact"]
        return any(indicator in context_lower for indicator in name_indicators)

    def _is_location_context(self, context: str) -> bool:
        """Check if context indicates a location."""
        context_lower = context.lower()
        location_indicators = [
            "live",
            "located",
            "based",
            "address",
            "city",
            "state",
            "country",
        ]
        return any(indicator in context_lower for indicator in location_indicators)

    def _is_organization_context(self, context: str) -> bool:
        """Check if context indicates an organization."""
        context_lower = context.lower()
        org_indicators = ["work", "company", "office", "job", "career", "business"]
        return any(indicator in context_lower for indicator in org_indicators)

    def _is_job_title_context(self, context: str) -> bool:
        """Check if context indicates a job title."""
        context_lower = context.lower()
        job_indicators = ["work", "job", "position", "role", "career", "title"]
        return any(indicator in context_lower for indicator in job_indicators)

    def _classify_location(self, location: str) -> str:
        """Classify location type."""
        if location in self.dictionaries["cities"]:
            return "city"
        elif location in self.dictionaries["states"]:
            return "state"
        elif location in self.dictionaries["countries"]:
            return "country"
        else:
            return "unknown"

    def _get_context(
        self, text: str, start: int, end: int, window_size: int = 50
    ) -> str:
        """Get context around a match."""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]

    def add_dictionary_entry(self, pii_type: str, entry: str) -> None:
        """Add an entry to a dictionary."""
        if pii_type == "first_names":
            self.dictionaries["first_names"].add(entry.lower())
        elif pii_type == "last_names":
            self.dictionaries["last_names"].add(entry.lower())
        elif pii_type == "companies":
            self.dictionaries["companies"].add(entry.lower())
        elif pii_type == "cities":
            self.dictionaries["cities"].add(entry.lower())

    def add_context_keyword(self, pii_type: str, keyword: str) -> None:
        """Add a context keyword."""
        if pii_type not in self.context_keywords:
            self.context_keywords[pii_type] = []
        self.context_keywords[pii_type].append(keyword.lower())
