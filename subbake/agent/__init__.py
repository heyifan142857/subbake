"""SubBake agent package.

All public agent symbols are re-exported here so that existing imports
from ``subbake.agent`` continue to work.
"""

from __future__ import annotations

# -- core agent class and constants ----------------------------------------
from ._core import (  # noqa: I001
    AGENT_LOOP_MAX_STEPS,
    CONFIG_BOOTSTRAP_CREATE,
    CONFIG_BOOTSTRAP_SKIP,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_MIN_OBSERVATIONS,
    NEW_PROFILE_VALUE,
    PROFILE_API_KEY_ENV_OPTIONS,
    PROFILE_PROVIDER_OPTIONS,
    PROFILE_TARGET_LANGUAGE_OPTIONS,
    SubBakeAgent,
    start_interactive_agent,
)

# -- loop / data structures -------------------------------------------------
from .loop import (  # noqa: I001
    DISCOVERY_TOOL_NAMES,
    GENERATED_SUBTITLE_MARKERS,
    MEDIA_SUFFIXES,
    MUTATING_TOOL_NAMES,
    PROTECTED_PATH_PARTS,
    STOPWORDS,
    AgentLoopState,
    AgentLoopStep,
    AgentObservation,
    FileCandidate,
    classify_candidate_path,
    executable_subtitle_path,
    format_candidate_lines,
    rank_file_candidates,
    strong_subtitle_candidates,
)

# -- UI helpers --------------------------------------------------------------
from .ui import (  # noqa: I001
    print_file_completion,
    print_file_op_result,
    print_help,
    print_series_completion,
    print_series_summary,
    print_tool_call_preview,
    print_translation_start,
    render_mode_label,
)

# -- session -----------------------------------------------------------------
from .session import (  # noqa: I001
    AgentSession,
    AgentSessionStore,
    SESSION_VERSION,
)

# -- argument parsing --------------------------------------------------------
from .arg_parser import (  # noqa: I001
    REFERENCE_RE as _REFERENCE_RE_ARG_PARSER,
    arguments_with_text_overrides,
    bilingual_requested,
    bool_argument,
    language_phrases,
    line_without_output_format_phrases,
    monolingual_requested,
    output_format_from_argument,
    output_format_from_text,
    output_format_patterns,
    resolve_user_path,
    series_suffixes_from_argument,
    series_suffixes_from_text,
    source_language_from_text,
    target_language_for_bilingual_pair,
    target_language_from_text,
    title_tokens_from_text,
    translation_arguments_from_text,
    translation_values_for_tool,
)

# -- tool registry ----------------------------------------------------------
from .tool_registry import (  # noqa: I001
    ALL_TOOL_SPECS,
    ALWAYS_AVAILABLE_TOOLS,
    TOOL_CATEGORIES,
    build_tool_specs,
)

