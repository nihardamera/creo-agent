"""The shared vocabulary: graph nodes, allowlisted actions, final states.

The dataset's labels and the agent both use these names, but nothing else:
labels are written from each scenario's intent, never by running the agent.
"""

from enum import StrEnum


class Node(StrEnum):
    """Every node the workflow graph can visit, in roughly the order it visits them."""

    INGEST = "ingest"
    SCREEN = "screen"
    DEDUPE = "dedupe"
    TICKET = "ticket"
    GATHER_CONTEXT = "gather_context"
    SELECT_RUNBOOK = "select_runbook"
    DIAGNOSTICS = "diagnostics"
    CLASSIFY = "classify"
    DECIDE_WAIT = "decide_wait"
    WAIT = "wait"
    RECHECK = "recheck"
    PROPOSE_FIX = "propose_fix"
    APPROVAL = "approval"
    EXECUTE = "execute"
    VERIFY = "verify"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    FINALIZE = "finalize"


# The only nodes that may call the model. Everything else is deterministic.
LLM_NODES: frozenset[Node] = frozenset({Node.CLASSIFY, Node.ESCALATE})


class Action(StrEnum):
    """The allowlist. The executor refuses any action not named here."""

    RESTART_WINDOWS_SERVICE = "restart_windows_service"
    START_APP_POOL = "start_app_pool"
    RECYCLE_APP_POOL = "recycle_app_pool"
    RESTART_IIS = "restart_iis"
    CLEAR_TEMP_FILES = "clear_temp_files"
    ROTATE_IIS_LOGS = "rotate_iis_logs"
    CLEAR_WU_CACHE = "clear_wu_cache"
    KILL_PROCESS = "kill_process"
    RERUN_SCHEDULED_TASK = "rerun_scheduled_task"
    FLUSH_DNS_CACHE = "flush_dns_cache"
    RESTART_SYSTEMD_UNIT = "restart_systemd_unit"
    VACUUM_JOURNAL = "vacuum_journal"
    CLEAN_PACKAGE_CACHE = "clean_package_cache"
    RESTART_QUEUE_CONSUMER = "restart_queue_consumer"


class Diagnostic(StrEnum):
    """Read-only checks the runbooks may run. None of them changes a host."""

    GET_SERVICE_STATUS = "get_service_status"
    GET_APP_POOL_STATE = "get_app_pool_state"
    GET_RECENT_EVENTS = "get_recent_events"
    GET_DISK_USAGE = "get_disk_usage"
    GET_LARGE_DIRS = "get_large_dirs"
    GET_TOP_PROCESSES = "get_top_processes"
    HTTP_PROBE = "http_probe"
    GET_TASK_HISTORY = "get_task_history"
    GET_CERT_INFO = "get_cert_info"
    PING_HOST = "ping_host"
    GET_QUEUE_STATS = "get_queue_stats"
    GET_SYSTEMD_STATUS = "get_systemd_status"
    GET_JOURNAL_TAIL = "get_journal_tail"
    GET_SQL_BLOCKING = "get_sql_blocking"
    RESOLVE_NAME = "resolve_name"


class FinalState(StrEnum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    DUPLICATE = "duplicate"  # same monitor and host as an open run; linked to its ticket
    CORRELATED = "correlated"  # part of an open incident on the same service; linked
    IGNORED = "ignored"  # stale, Warn/No Data, or a Recovered with no open run
    REJECTED = "rejected"  # refused at ingress: never became a run


class Transition(StrEnum):
    """Datadog's $ALERT_TRANSITION values."""

    TRIGGERED = "Triggered"
    RE_TRIGGERED = "Re-Triggered"
    RECOVERED = "Recovered"
    WARN = "Warn"
    RE_WARN = "Re-Warn"
    NO_DATA = "No Data"
    RE_NO_DATA = "Re-No Data"
    RENOTIFY = "Renotify"
