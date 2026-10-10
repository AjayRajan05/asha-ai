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

"""
LlamaIndex Integration for ASHA

This module provides direct integration with LlamaIndex components,
allowing ASHA to secure and optimize queries and prompts within
LlamaIndex pipelines.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from llama_index.core import QueryEngine, VectorStoreIndex
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.llms import LLM, CustomLLM
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.schema import QueryBundle, Document

    LLAMAINDEX_AVAILABLE = True
else:
    try:
        from llama_index.core import QueryEngine, VectorStoreIndex
        from llama_index.core.base.base_query_engine import BaseQueryEngine
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.llms import LLM, CustomLLM
        from llama_index.core.prompts import PromptTemplate
        from llama_index.core.schema import QueryBundle, Document

        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        LLAMAINDEX_AVAILABLE = False

        class QueryEngine:
            pass

        class VectorStoreIndex:
            @classmethod
            def from_documents(cls, *args: Any, **kwargs: Any) -> "VectorStoreIndex":
                return cls()

        class BaseQueryEngine:
            def query(self, query_bundle: Any) -> Any:
                return None

            async def aquery(self, query_bundle: Any) -> Any:
                return None

        class BaseRetriever:
            def retrieve(self, query_bundle: Any) -> List[Any]:
                return []

            async def aretrieve(self, query_bundle: Any) -> List[Any]:
                return []

        class LLM:
            model_name: str = "llm"

            def complete(self, prompt: str, **kwargs: Any) -> str:
                return prompt

        class CustomLLM:
            pass

        class PromptTemplate:
            def __init__(self, template: str, **kwargs: Any) -> None:
                self.template = template

            def format(self, **kwargs: Any) -> str:
                return self.template

        class QueryBundle:
            def __init__(
                self,
                query_str: str,
                custom_embedding_strs: Any = None,
                embedding_strs: Any = None,
            ) -> None:
                self.query_str = query_str
                self.custom_embedding_strs = custom_embedding_strs
                self.embedding_strs = embedding_strs

        class Document:
            def __init__(
                self,
                text: str,
                metadata: Optional[Dict[str, Any]] = None,
                id_: Optional[str] = None,
            ) -> None:
                self.text = text
                self.metadata = metadata or {}
                self.id_ = id_


from ...utils.dropin import process, _coerce_process_output


def _optimize_prompt_text(
    text: str,
    *,
    mode: str = "balanced",
    token_budget: int,
    debug_metrics: bool,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Run process() and return optimized text plus optional metrics."""
    if debug_metrics:
        processed = process(
            text,
            mode=mode,
            token_budget=token_budget,
        )
        result = processed.to_dict()
        return str(result["optimized"]), result
    return (
        _coerce_process_output(
            process(text, mode=mode, token_budget=token_budget),
            text,
        ),
        None,
    )


class ASHAPromptTemplate(PromptTemplate):
    """
    LlamaIndex PromptTemplate wrapper that applies ASHA optimization.

    This wrapper automatically processes prompts through ASHA's
    security and optimization pipeline before using them in LlamaIndex.
    """

    def __init__(
        self,
        template: str,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ASHA-enhanced prompt template.

        Args:
            template: Prompt template string
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Install with: pip install llama-index\n"
                "Or use the standalone ASHA functions instead."
            )

        super().__init__(template=template, **kwargs)
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self._last_metrics: Optional[Dict[str, Any]] = None

    def format(self, **kwargs: Any) -> str:
        """
        Format the template with input variables and apply ASHA optimization.

        Args:
            **kwargs: Input variable values

        Returns:
            Optimized prompt string
        """
        # Format the template normally
        formatted_prompt = super().format(**kwargs)

        # Apply ASHA optimization
        mode = "balanced" if self.privacy else "off"
        if self.debug_metrics:
            processed = process(
                formatted_prompt,
                mode=mode,
                token_budget=self.token_budget,
            )
            result = processed.to_dict()
            self._last_metrics = result
            return str(result["optimized"])
        return _coerce_process_output(
            process(
                formatted_prompt,
                mode=mode,
                token_budget=self.token_budget,
            ),
            formatted_prompt,
        )

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last prompt processing."""
        return self._last_metrics


class ASHALLM(CustomLLM):
    """
    LlamaIndex LLM wrapper that applies ASHA security and optimization.

    This wrapper intercepts all prompts sent to the LLM and processes them
    through ASHA's pipeline before forwarding to the actual LLM.
    """

    context_window: int = 4096
    num_output: int = 256
    model_name: str = "asha_llm"

    def __init__(
        self,
        llm: LLM,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize ASHA-enhanced LLM wrapper.

        Args:
            llm: Base LLM to wrap
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self._last_metrics: Optional[Dict[str, Any]] = None
        self.model_name = f"asha_{llm.model_name}"

    def _get_prompt_type(self, prompt: str) -> str:
        """Determine prompt type for specialized processing."""
        prompt_lower = prompt.lower()
        if "query" in prompt_lower or "search" in prompt_lower:
            return "query"
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return "summary"
        elif "analyze" in prompt_lower or "analysis" in prompt_lower:
            return "analysis"
        else:
            return "general"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Process prompt through ASHA and forward to LLM.

        Args:
            prompt: Input prompt
            **kwargs: Additional completion arguments

        Returns:
            LLM response
        """
        # Apply ASHA optimization
        mode = "balanced" if self.privacy else "off"
        if self.debug_metrics:
            processed = process(
                prompt,
                mode=mode,
                token_budget=self.token_budget,
            )
            result = processed.to_dict()
            self._last_metrics = result
            optimized_prompt = str(result["optimized"])
        else:
            optimized_prompt = _coerce_process_output(
                process(
                    prompt,
                    mode=("balanced" if self.privacy else "off"),
                    token_budget=self.token_budget,
                ),
                prompt,
            )

        # Forward to actual LLM
        return str(self.llm.complete(optimized_prompt, **kwargs))

    def complete_request(self, prompt: str, **kwargs: Any) -> str:
        """Process complete request with ASHA optimization."""
        return self.complete(prompt, **kwargs)

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last prompt processing."""
        return self._last_metrics


class ASHAQueryEngine(BaseQueryEngine):
    """
    LlamaIndex QueryEngine wrapper that applies ASHA optimization.

    This wrapper processes queries through ASHA's security and
    optimization pipeline before sending them to the underlying query engine.
    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
        optimize_queries: bool = True,
        optimize_retrieval: bool = True,
    ) -> None:
        """
        Initialize ASHA-enhanced query engine.

        Args:
            query_engine: Base query engine to wrap
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
            optimize_queries: Optimize query prompts
            optimize_retrieval: Optimize retrieval queries
        """
        self._query_engine = query_engine
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self.optimize_queries = optimize_queries
        self.optimize_retrieval = optimize_retrieval
        self._last_metrics: Optional[Dict[str, Any]] = None

    def query(self, query_bundle: QueryBundle) -> Any:
        """
        Process query through ASHA and forward to query engine.

        Args:
            query_bundle: Query bundle containing the query

        Returns:
            Query response
        """
        query_str = query_bundle.query_str

        # Apply ASHA optimization to query
        if self.optimize_queries:
            optimized_query, metrics = _optimize_prompt_text(
                query_str,
                mode=("balanced" if self.privacy else "off"),
                token_budget=self.token_budget,
                debug_metrics=self.debug_metrics,
            )
            if metrics is not None:
                self._last_metrics = metrics

            # Create new query bundle with optimized query
            optimized_bundle = QueryBundle(
                query_str=optimized_query,
                custom_embedding_strs=query_bundle.custom_embedding_strs,
                embedding_strs=query_bundle.embedding_strs,
            )
        else:
            optimized_bundle = query_bundle

        # Forward to actual query engine
        return self._query_engine.query(optimized_bundle)

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last query processing."""
        return self._last_metrics

    async def aquery(self, query_bundle: QueryBundle) -> Any:
        """Async query with ASHA optimization."""
        query_str = query_bundle.query_str

        # Apply ASHA optimization to query
        if self.optimize_queries:
            optimized_query, metrics = _optimize_prompt_text(
                query_str,
                mode=("balanced" if self.privacy else "off"),
                token_budget=self.token_budget,
                debug_metrics=self.debug_metrics,
            )
            if metrics is not None:
                self._last_metrics = metrics

            # Create new query bundle with optimized query
            optimized_bundle = QueryBundle(
                query_str=optimized_query,
                custom_embedding_strs=query_bundle.custom_embedding_strs,
                embedding_strs=query_bundle.embedding_strs,
            )
        else:
            optimized_bundle = query_bundle

        # Forward to actual query engine
        return await self._query_engine.aquery(optimized_bundle)


class ASHARetriever(BaseRetriever):
    """
    LlamaIndex Retriever wrapper that applies ASHA optimization.

    This wrapper processes retrieval queries through ASHA's
    optimization pipeline before sending them to the underlying retriever.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
    ) -> None:
        """
        Initialize ASHA-enhanced retriever.

        Args:
            retriever: Base retriever to wrap
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        self._retriever = retriever
        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self._last_metrics: Optional[Dict[str, Any]] = None

    def retrieve(self, query_bundle: QueryBundle) -> List[Document]:
        """
        Process query through ASHA and forward to retriever.

        Args:
            query_bundle: Query bundle containing the query

        Returns:
            Retrieved documents
        """
        query_str = query_bundle.query_str

        optimized_query, metrics = _optimize_prompt_text(
            query_str,
            mode=("balanced" if self.privacy else "off"),
            token_budget=self.token_budget,
            debug_metrics=self.debug_metrics,
        )
        if metrics is not None:
            self._last_metrics = metrics

        # Create new query bundle with optimized query
        optimized_bundle = QueryBundle(
            query_str=optimized_query,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
            embedding_strs=query_bundle.embedding_strs,
        )

        # Forward to actual retriever
        return cast(List[Document], self._retriever.retrieve(optimized_bundle))

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from the last retrieval processing."""
        return self._last_metrics

    async def aretrieve(self, query_bundle: QueryBundle) -> List[Document]:
        """Async retrieval with ASHA optimization."""
        query_str = query_bundle.query_str

        optimized_query, metrics = _optimize_prompt_text(
            query_str,
            mode=("balanced" if self.privacy else "off"),
            token_budget=self.token_budget,
            debug_metrics=self.debug_metrics,
        )
        if metrics is not None:
            self._last_metrics = metrics

        # Create new query bundle with optimized query
        optimized_bundle = QueryBundle(
            query_str=optimized_query,
            custom_embedding_strs=query_bundle.custom_embedding_strs,
            embedding_strs=query_bundle.embedding_strs,
        )

        # Forward to actual retriever
        return cast(List[Document], await self._retriever.aretrieve(optimized_bundle))


# Convenience functions for easy integration
def wrap_prompt_template(
    template: str,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
) -> ASHAPromptTemplate:
    """
    Create a ASHA-enhanced prompt template.

    Args:
        template: Prompt template string
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Returns:
        ASHA-enhanced prompt template
    """
    return ASHAPromptTemplate(
        template=template,
        privacy=privacy,
        token_budget=token_budget,
        debug_metrics=debug_metrics,
    )


def wrap_llm(
    llm: LLM,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
) -> ASHALLM:
    """
    Wrap an LLM with ASHA optimization.

    Args:
        llm: LLM to wrap
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Returns:
        ASHA-enhanced LLM
    """
    return ASHALLM(
        llm=llm, privacy=privacy, token_budget=token_budget, debug_metrics=debug_metrics
    )


def wrap_query_engine(
    query_engine: BaseQueryEngine,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
    optimize_queries: bool = True,
    optimize_retrieval: bool = True,
) -> ASHAQueryEngine:
    """
    Wrap a query engine with ASHA optimization.

    Args:
        query_engine: Query engine to wrap
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics
        optimize_queries: Optimize query prompts
        optimize_retrieval: Optimize retrieval queries

    Returns:
        ASHA-enhanced query engine
    """
    return ASHAQueryEngine(
        query_engine=query_engine,
        privacy=privacy,
        token_budget=token_budget,
        debug_metrics=debug_metrics,
        optimize_queries=optimize_queries,
        optimize_retrieval=optimize_retrieval,
    )


def wrap_retriever(
    retriever: BaseRetriever,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
) -> ASHARetriever:
    """
    Wrap a retriever with ASHA optimization.

    Args:
        retriever: Retriever to wrap
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics

    Returns:
        ASHA-enhanced retriever
    """
    return ASHARetriever(
        retriever=retriever,
        privacy=privacy,
        token_budget=token_budget,
        debug_metrics=debug_metrics,
    )


class ASHAPostProcessor:
    """
    LlamaIndex post-processor that applies ASHA security and optimization
    to query results and responses.

    This post-processor ensures that all text returned from LlamaIndex
    pipelines is properly secured and optimized.
    """

    def __init__(
        self,
        privacy: bool = True,
        token_budget: int = 1200,
        debug_metrics: bool = False,
    ) -> None:
        """
        Initialize ASHA post-processor.

        Args:
            privacy: Enable privacy features
            token_budget: Token budget for optimization
            debug_metrics: Return optimization metrics
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Install with: pip install llama-index\n"
                "Or use the standalone ASHA functions instead."
            )

        self.privacy = privacy
        self.token_budget = token_budget
        self.debug_metrics = debug_metrics
        self._last_metrics: Optional[Dict[str, Any]] = None

    def postprocess_nodes(self, nodes: List[Any], **kwargs: Any) -> List[Any]:
        """
        Apply ASHA processing to document nodes.

        Args:
            nodes: List of document nodes to process
            **kwargs: Additional arguments

        Returns:
            List of processed document nodes
        """
        processed_nodes: List[Any] = []

        for node in nodes:
            if hasattr(node, "text") and node.text:
                # Process the node text through ASHA
                processed = process(
                    node.text,
                    mode=("balanced" if self.privacy else "off"),
                    token_budget=self.token_budget,
                )
                result = processed.to_dict() if self.debug_metrics else None
                processed_text = processed.output

                if self.debug_metrics and result:
                    self._last_metrics = result.get("metrics", {})

                # Create new node with processed text
                if hasattr(node, "copy"):
                    processed_node = node.copy()
                    processed_node.text = processed_text
                else:
                    # Fallback for different node types
                    processed_node = Document(
                        text=processed_text,
                        metadata=getattr(node, "metadata", {}),
                        id_=getattr(node, "id_", None),
                    )

                processed_nodes.append(processed_node)
            else:
                processed_nodes.append(node)

        return processed_nodes

    def postprocess_query(self, query_str: str, **kwargs: Any) -> str:
        """
        Apply ASHA processing to query string.

        Args:
            query_str: Query string to process
            **kwargs: Additional arguments

        Returns:
            Processed query string
        """
        processed = process(
            query_str,
            mode=("balanced" if self.privacy else "off"),
            token_budget=self.token_budget,
        )

        if self.debug_metrics:
            result = processed.to_dict()
            self._last_metrics = cast(Dict[str, Any], result.get("metrics", {}))
        return processed.output

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the last processing metrics.

        Returns:
            Dictionary of processing metrics
        """
        return self._last_metrics or {}


def create_asha_index(
    documents: List[Document],
    llm: LLM,
    privacy: bool = True,
    token_budget: int = 1200,
    debug_metrics: bool = False,
    **index_kwargs: Any,
) -> VectorStoreIndex:
    """
    Create a ASHA-enhanced VectorStoreIndex.

    Args:
        documents: Documents to index
        llm: LLM to use with ASHA optimization
        privacy: Enable privacy features
        token_budget: Token budget for optimization
        debug_metrics: Return optimization metrics
        **index_kwargs: Additional index arguments

    Returns:
        ASHA-enhanced VectorStoreIndex
    """
    # Wrap the LLM with ASHA
    asha_llm = wrap_llm(llm, privacy, token_budget, debug_metrics)

    # Create index with ASHA-enhanced LLM
    index = VectorStoreIndex.from_documents(documents, llm=asha_llm, **index_kwargs)

    return cast(VectorStoreIndex, index)
