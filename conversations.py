from typing import Any, Optional, List, Union, Dict, TypeVar, Type, cast, Callable
from enum import Enum
from uuid import UUID
from datetime import datetime
import dateutil.parser


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


class AuthorMetadata:
    pass

    def __init__(self, ) -> None:
        pass

    @staticmethod
    def from_dict(obj: Any) -> 'AuthorMetadata':
        assert isinstance(obj, dict)
        return AuthorMetadata()

    def to_dict(self) -> dict:
        result: dict = {}
        return result


class Recipient(Enum):
    ALL = "all"
    BROWSER = "browser"
    PYTHON = "python"


class Role(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class Author:
    role: Role
    name: Optional[Recipient]
    metadata: AuthorMetadata

    def __init__(self, role: Role, name: Optional[Recipient], metadata: AuthorMetadata) -> None:
        self.role = role
        self.name = name
        self.metadata = metadata

    @staticmethod
    def from_dict(obj: Any) -> 'Author':
        assert isinstance(obj, dict)
        role = Role(obj.get("role"))
        name = from_union([from_none, Recipient], obj.get("name"))
        metadata = AuthorMetadata.from_dict(obj.get("metadata"))
        return Author(role, name, metadata)

    def to_dict(self) -> dict:
        result: dict = {}
        result["role"] = to_enum(Role, self.role)
        result["name"] = from_union([from_none, lambda x: to_enum(Recipient, x)], self.name)
        result["metadata"] = to_class(AuthorMetadata, self.metadata)
        return result


class ContentType(Enum):
    CODE = "code"
    EXECUTION_OUTPUT = "execution_output"
    SYSTEM_ERROR = "system_error"
    TETHER_BROWSING_DISPLAY = "tether_browsing_display"
    TETHER_QUOTE = "tether_quote"
    TEXT = "text"


class Language(Enum):
    UNKNOWN = "unknown"


class MessageContent:
    content_type: ContentType
    parts: Optional[List[str]]
    language: Optional[Language]
    text: Optional[str]
    result: Optional[str]
    summary: Optional[str]
    url: Optional[str]
    domain: Optional[str]
    title: Optional[str]
    name: Optional[str]

    def __init__(self, content_type: ContentType, parts: Optional[List[str]], language: Optional[Language], text: Optional[str], result: Optional[str], summary: Optional[str], url: Optional[str], domain: Optional[str], title: Optional[str], name: Optional[str]) -> None:
        self.content_type = content_type
        self.parts = parts
        self.language = language
        self.text = text
        self.result = result
        self.summary = summary
        self.url = url
        self.domain = domain
        self.title = title
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> 'MessageContent':
        assert isinstance(obj, dict)
        content_type = ContentType(obj.get("content_type"))
        parts = from_union([lambda x: from_list(from_str, x), from_none], obj.get("parts"))
        language = from_union([Language, from_none], obj.get("language"))
        text = from_union([from_str, from_none], obj.get("text"))
        result = from_union([from_str, from_none], obj.get("result"))
        summary = from_union([from_none, from_str], obj.get("summary"))
        url = from_union([from_str, from_none], obj.get("url"))
        domain = from_union([from_str, from_none], obj.get("domain"))
        title = from_union([from_str, from_none], obj.get("title"))
        name = from_union([from_str, from_none], obj.get("name"))
        return MessageContent(content_type, parts, language, text, result, summary, url, domain, title, name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["content_type"] = to_enum(ContentType, self.content_type)
        if self.parts is not None:
            result["parts"] = from_union([lambda x: from_list(from_str, x), from_none], self.parts)
        if self.language is not None:
            result["language"] = from_union([lambda x: to_enum(Language, x), from_none], self.language)
        if self.text is not None:
            result["text"] = from_union([from_str, from_none], self.text)
        if self.result is not None:
            result["result"] = from_union([from_str, from_none], self.result)
        if self.summary is not None:
            result["summary"] = from_union([from_none, from_str], self.summary)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        if self.domain is not None:
            result["domain"] = from_union([from_str, from_none], self.domain)
        if self.title is not None:
            result["title"] = from_union([from_str, from_none], self.title)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        return result


class Data:
    text_plain: str
    text_html: Optional[str]
    image_vnd_openai_fileservice_png: Optional[str]

    def __init__(self, text_plain: str, text_html: Optional[str], image_vnd_openai_fileservice_png: Optional[str]) -> None:
        self.text_plain = text_plain
        self.text_html = text_html
        self.image_vnd_openai_fileservice_png = image_vnd_openai_fileservice_png

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        assert isinstance(obj, dict)
        text_plain = from_str(obj.get("text/plain"))
        text_html = from_union([from_str, from_none], obj.get("text/html"))
        image_vnd_openai_fileservice_png = from_union([from_str, from_none], obj.get("image/vnd.openai.fileservice.png"))
        return Data(text_plain, text_html, image_vnd_openai_fileservice_png)

    def to_dict(self) -> dict:
        result: dict = {}
        result["text/plain"] = from_str(self.text_plain)
        if self.text_html is not None:
            result["text/html"] = from_union([from_str, from_none], self.text_html)
        if self.image_vnd_openai_fileservice_png is not None:
            result["image/vnd.openai.fileservice.png"] = from_union([from_str, from_none], self.image_vnd_openai_fileservice_png)
        return result


class ExecutionState(Enum):
    BUSY = "busy"
    IDLE = "idle"


class JupyterMessageContent:
    execution_state: Optional[ExecutionState]
    data: Optional[Data]
    name: Optional[str]
    text: Optional[str]

    def __init__(self, execution_state: Optional[ExecutionState], data: Optional[Data], name: Optional[str], text: Optional[str]) -> None:
        self.execution_state = execution_state
        self.data = data
        self.name = name
        self.text = text

    @staticmethod
    def from_dict(obj: Any) -> 'JupyterMessageContent':
        assert isinstance(obj, dict)
        execution_state = from_union([ExecutionState, from_none], obj.get("execution_state"))
        data = from_union([Data.from_dict, from_none], obj.get("data"))
        name = from_union([from_str, from_none], obj.get("name"))
        text = from_union([from_str, from_none], obj.get("text"))
        return JupyterMessageContent(execution_state, data, name, text)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.execution_state is not None:
            result["execution_state"] = from_union([lambda x: to_enum(ExecutionState, x), from_none], self.execution_state)
        if self.data is not None:
            result["data"] = from_union([lambda x: to_class(Data, x), from_none], self.data)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.text is not None:
            result["text"] = from_union([from_str, from_none], self.text)
        return result


class MsgType(Enum):
    DISPLAY_DATA = "display_data"
    EXECUTE_INPUT = "execute_input"
    EXECUTE_RESULT = "execute_result"
    STATUS = "status"
    STREAM = "stream"


class MsgID(Enum):
    A0895343_028_FFB2177_B94271_C28_BBCAB_2_1 = "a0895343-028ffb2177b94271c28bbcab_2_1"
    DFDF6_B94_FD2738_BD091756_D372_BED48_E_2_1 = "dfdf6b94-fd2738bd091756d372bed48e_2_1"
    E2_CEFA71_EC236_F78831_DA77_F64_E3_F38_C_2_1 = "e2cefa71-ec236f78831da77f64e3f38c_2_1"
    THE_19_BF9_B10_848_B1_EB8015_BB71571_AB8163_2_1 = "19bf9b10-848b1eb8015bb71571ab8163_2_1"
    THE_507_E7_CB6_B59338781836_D6_EF3_E1_F6792_2_1 = "507e7cb6-b59338781836d6ef3e1f6792_2_1"
    THE_7_B72_C3_A7_0_B9_D20_BE10_B1_AAF4_A7956205_2_1 = "7b72c3a7-0b9d20be10b1aaf4a7956205_2_1"
    THE_8_D1_B6081_BC9_EECCEFA3_C0955_DDA4_EFCF_2_1 = "8d1b6081-bc9eeccefa3c0955dda4efcf_2_1"


class ParentHeader:
    msg_id: MsgID
    version: str

    def __init__(self, msg_id: MsgID, version: str) -> None:
        self.msg_id = msg_id
        self.version = version

    @staticmethod
    def from_dict(obj: Any) -> 'ParentHeader':
        assert isinstance(obj, dict)
        msg_id = MsgID(obj.get("msg_id"))
        version = from_str(obj.get("version"))
        return ParentHeader(msg_id, version)

    def to_dict(self) -> dict:
        result: dict = {}
        result["msg_id"] = to_enum(MsgID, self.msg_id)
        result["version"] = from_str(self.version)
        return result


class JupyterMessage:
    msg_type: MsgType
    parent_header: ParentHeader
    content: Optional[JupyterMessageContent]

    def __init__(self, msg_type: MsgType, parent_header: ParentHeader, content: Optional[JupyterMessageContent]) -> None:
        self.msg_type = msg_type
        self.parent_header = parent_header
        self.content = content

    @staticmethod
    def from_dict(obj: Any) -> 'JupyterMessage':
        assert isinstance(obj, dict)
        msg_type = MsgType(obj.get("msg_type"))
        parent_header = ParentHeader.from_dict(obj.get("parent_header"))
        content = from_union([JupyterMessageContent.from_dict, from_none], obj.get("content"))
        return JupyterMessage(msg_type, parent_header, content)

    def to_dict(self) -> dict:
        result: dict = {}
        result["msg_type"] = to_enum(MsgType, self.msg_type)
        result["parent_header"] = to_class(ParentHeader, self.parent_header)
        if self.content is not None:
            result["content"] = from_union([lambda x: to_class(JupyterMessageContent, x), from_none], self.content)
        return result


class MessageElement:
    message_type: str
    time: float
    stream_name: Optional[str]
    sender: str
    text: Optional[str]
    image_payload: None
    image_url: Optional[str]

    def __init__(self, message_type: str, time: float, stream_name: Optional[str], sender: str, text: Optional[str], image_payload: None, image_url: Optional[str]) -> None:
        self.message_type = message_type
        self.time = time
        self.stream_name = stream_name
        self.sender = sender
        self.text = text
        self.image_payload = image_payload
        self.image_url = image_url

    @staticmethod
    def from_dict(obj: Any) -> 'MessageElement':
        assert isinstance(obj, dict)
        message_type = from_str(obj.get("message_type"))
        time = from_float(obj.get("time"))
        stream_name = from_union([from_str, from_none], obj.get("stream_name"))
        sender = from_str(obj.get("sender"))
        text = from_union([from_str, from_none], obj.get("text"))
        image_payload = from_none(obj.get("image_payload"))
        image_url = from_union([from_str, from_none], obj.get("image_url"))
        return MessageElement(message_type, time, stream_name, sender, text, image_payload, image_url)

    def to_dict(self) -> dict:
        result: dict = {}
        result["message_type"] = from_str(self.message_type)
        result["time"] = to_float(self.time)
        if self.stream_name is not None:
            result["stream_name"] = from_union([from_str, from_none], self.stream_name)
        result["sender"] = from_str(self.sender)
        if self.text is not None:
            result["text"] = from_union([from_str, from_none], self.text)
        if self.image_payload is not None:
            result["image_payload"] = from_none(self.image_payload)
        if self.image_url is not None:
            result["image_url"] = from_union([from_str, from_none], self.image_url)
        return result


class AggregateResultStatus(Enum):
    SUCCESS = "success"


class AggregateResult:
    status: AggregateResultStatus
    run_id: UUID
    start_time: float
    update_time: float
    code: str
    end_time: float
    final_expression_output: Optional[str]
    in_kernel_exception: None
    system_exception: None
    messages: List[MessageElement]
    jupyter_messages: List[JupyterMessage]
    timeout_triggered: None

    def __init__(self, status: AggregateResultStatus, run_id: UUID, start_time: float, update_time: float, code: str, end_time: float, final_expression_output: Optional[str], in_kernel_exception: None, system_exception: None, messages: List[MessageElement], jupyter_messages: List[JupyterMessage], timeout_triggered: None) -> None:
        self.status = status
        self.run_id = run_id
        self.start_time = start_time
        self.update_time = update_time
        self.code = code
        self.end_time = end_time
        self.final_expression_output = final_expression_output
        self.in_kernel_exception = in_kernel_exception
        self.system_exception = system_exception
        self.messages = messages
        self.jupyter_messages = jupyter_messages
        self.timeout_triggered = timeout_triggered

    @staticmethod
    def from_dict(obj: Any) -> 'AggregateResult':
        assert isinstance(obj, dict)
        status = AggregateResultStatus(obj.get("status"))
        run_id = UUID(obj.get("run_id"))
        start_time = from_float(obj.get("start_time"))
        update_time = from_float(obj.get("update_time"))
        code = from_str(obj.get("code"))
        end_time = from_float(obj.get("end_time"))
        final_expression_output = from_union([from_none, from_str], obj.get("final_expression_output"))
        in_kernel_exception = from_none(obj.get("in_kernel_exception"))
        system_exception = from_none(obj.get("system_exception"))
        messages = from_list(MessageElement.from_dict, obj.get("messages"))
        jupyter_messages = from_list(JupyterMessage.from_dict, obj.get("jupyter_messages"))
        timeout_triggered = from_none(obj.get("timeout_triggered"))
        return AggregateResult(status, run_id, start_time, update_time, code, end_time, final_expression_output, in_kernel_exception, system_exception, messages, jupyter_messages, timeout_triggered)

    def to_dict(self) -> dict:
        result: dict = {}
        result["status"] = to_enum(AggregateResultStatus, self.status)
        result["run_id"] = str(self.run_id)
        result["start_time"] = to_float(self.start_time)
        result["update_time"] = to_float(self.update_time)
        result["code"] = from_str(self.code)
        result["end_time"] = to_float(self.end_time)
        result["final_expression_output"] = from_union([from_none, from_str], self.final_expression_output)
        result["in_kernel_exception"] = from_none(self.in_kernel_exception)
        result["system_exception"] = from_none(self.system_exception)
        result["messages"] = from_list(lambda x: to_class(MessageElement, x), self.messages)
        result["jupyter_messages"] = from_list(lambda x: to_class(JupyterMessage, x), self.jupyter_messages)
        result["timeout_triggered"] = from_none(self.timeout_triggered)
        return result


class Attachment:
    name: str
    url: str

    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url

    @staticmethod
    def from_dict(obj: Any) -> 'Attachment':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        url = from_str(obj.get("url"))
        return Attachment(name, url)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = from_str(self.name)
        result["url"] = from_str(self.url)
        return result


class MetadataListElement:
    title: str
    url: str
    text: str
    pub_date: Optional[datetime]

    def __init__(self, title: str, url: str, text: str, pub_date: Optional[datetime]) -> None:
        self.title = title
        self.url = url
        self.text = text
        self.pub_date = pub_date

    @staticmethod
    def from_dict(obj: Any) -> 'MetadataListElement':
        assert isinstance(obj, dict)
        title = from_str(obj.get("title"))
        url = from_str(obj.get("url"))
        text = from_str(obj.get("text"))
        pub_date = from_union([from_none, from_datetime], obj.get("pub_date"))
        return MetadataListElement(title, url, text, pub_date)

    def to_dict(self) -> dict:
        result: dict = {}
        result["title"] = from_str(self.title)
        result["url"] = from_str(self.url)
        result["text"] = from_str(self.text)
        result["pub_date"] = from_union([from_none, lambda x: x.isoformat()], self.pub_date)
        return result


class Citation:
    start_ix: int
    end_ix: int
    metadata: MetadataListElement

    def __init__(self, start_ix: int, end_ix: int, metadata: MetadataListElement) -> None:
        self.start_ix = start_ix
        self.end_ix = end_ix
        self.metadata = metadata

    @staticmethod
    def from_dict(obj: Any) -> 'Citation':
        assert isinstance(obj, dict)
        start_ix = from_int(obj.get("start_ix"))
        end_ix = from_int(obj.get("end_ix"))
        metadata = MetadataListElement.from_dict(obj.get("metadata"))
        return Citation(start_ix, end_ix, metadata)

    def to_dict(self) -> dict:
        result: dict = {}
        result["start_ix"] = from_int(self.start_ix)
        result["end_ix"] = from_int(self.end_ix)
        result["metadata"] = to_class(MetadataListElement, self.metadata)
        return result


class Name(Enum):
    TETHER_OG = "tether_og"


class CitationFormat:
    name: Name

    def __init__(self, name: Name) -> None:
        self.name = name

    @staticmethod
    def from_dict(obj: Any) -> 'CitationFormat':
        assert isinstance(obj, dict)
        name = Name(obj.get("name"))
        return CitationFormat(name)

    def to_dict(self) -> dict:
        result: dict = {}
        result["name"] = to_enum(Name, self.name)
        return result


class CiteMetadata:
    citation_format: CitationFormat
    metadata_list: List[MetadataListElement]
    original_query: None

    def __init__(self, citation_format: CitationFormat, metadata_list: List[MetadataListElement], original_query: None) -> None:
        self.citation_format = citation_format
        self.metadata_list = metadata_list
        self.original_query = original_query

    @staticmethod
    def from_dict(obj: Any) -> 'CiteMetadata':
        assert isinstance(obj, dict)
        citation_format = CitationFormat.from_dict(obj.get("citation_format"))
        metadata_list = from_list(MetadataListElement.from_dict, obj.get("metadata_list"))
        original_query = from_none(obj.get("original_query"))
        return CiteMetadata(citation_format, metadata_list, original_query)

    def to_dict(self) -> dict:
        result: dict = {}
        result["citation_format"] = to_class(CitationFormat, self.citation_format)
        result["metadata_list"] = from_list(lambda x: to_class(MetadataListElement, x), self.metadata_list)
        result["original_query"] = from_none(self.original_query)
        return result


class Command(Enum):
    CLICK = "click"
    QUOTE = "quote"
    SCROLL = "scroll"
    SEARCH = "search"


class Stop(Enum):
    DIFF_MARKER = "<|diff_marker|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    IM_END = "<|im_end|>"


class TypeEnum(Enum):
    INTERRUPTED = "interrupted"
    MAX_TOKENS = "max_tokens"
    STOP = "stop"


class FinishDetails:
    type: TypeEnum
    stop_tokens: Optional[List[int]]
    stop: Optional[Stop]

    def __init__(self, type: TypeEnum, stop_tokens: Optional[List[int]], stop: Optional[Stop]) -> None:
        self.type = type
        self.stop_tokens = stop_tokens
        self.stop = stop

    @staticmethod
    def from_dict(obj: Any) -> 'FinishDetails':
        assert isinstance(obj, dict)
        type = TypeEnum(obj.get("type"))
        stop_tokens = from_union([lambda x: from_list(from_int, x), from_none], obj.get("stop_tokens"))
        stop = from_union([Stop, from_none], obj.get("stop"))
        return FinishDetails(type, stop_tokens, stop)

    def to_dict(self) -> dict:
        result: dict = {}
        result["type"] = to_enum(TypeEnum, self.type)
        if self.stop_tokens is not None:
            result["stop_tokens"] = from_union([lambda x: from_list(from_int, x), from_none], self.stop_tokens)
        if self.stop is not None:
            result["stop"] = from_union([lambda x: to_enum(Stop, x), from_none], self.stop)
        return result


class ModelSlug(Enum):
    GPT_4 = "gpt-4"
    GPT_4_BROWSING = "gpt-4-browsing"
    GPT_4_CODE_INTERPRETER = "gpt-4-code-interpreter"
    TEXT_DAVINCI_002_RENDER = "text-davinci-002-render"
    TEXT_DAVINCI_002_RENDER_SHA = "text-davinci-002-render-sha"


class MetadataStatus(Enum):
    FAILED = "failed"
    FINISHED = "finished"


class Timestamp(Enum):
    ABSOLUTE = "absolute"


class MessageMetadata:
    timestamp: Optional[Timestamp]
    message_type: None
    finish_details: Optional[FinishDetails]
    is_complete: Optional[bool]
    model_slug: Optional[ModelSlug]
    parent_id: Optional[UUID]
    aggregate_result: Optional[AggregateResult]
    attachments: Optional[List[Attachment]]
    cite_metadata: Optional[CiteMetadata]
    command: Optional[Command]
    args: Optional[List[Union[int, str]]]
    status: Optional[MetadataStatus]
    citations: Optional[List[Citation]]

    def __init__(self, timestamp: Optional[Timestamp], message_type: None, finish_details: Optional[FinishDetails], is_complete: Optional[bool], model_slug: Optional[ModelSlug], parent_id: Optional[UUID], aggregate_result: Optional[AggregateResult], attachments: Optional[List[Attachment]], cite_metadata: Optional[CiteMetadata], command: Optional[Command], args: Optional[List[Union[int, str]]], status: Optional[MetadataStatus], citations: Optional[List[Citation]]) -> None:
        self.timestamp = timestamp
        self.message_type = message_type
        self.finish_details = finish_details
        self.is_complete = is_complete
        self.model_slug = model_slug
        self.parent_id = parent_id
        self.aggregate_result = aggregate_result
        self.attachments = attachments
        self.cite_metadata = cite_metadata
        self.command = command
        self.args = args
        self.status = status
        self.citations = citations

    @staticmethod
    def from_dict(obj: Any) -> 'MessageMetadata':
        assert isinstance(obj, dict)
        timestamp = from_union([Timestamp, from_none], obj.get("timestamp_"))
        message_type = from_none(obj.get("message_type"))
        finish_details = from_union([FinishDetails.from_dict, from_none], obj.get("finish_details"))
        is_complete = from_union([from_bool, from_none], obj.get("is_complete"))
        model_slug = from_union([ModelSlug, from_none], obj.get("model_slug"))
        parent_id = from_union([lambda x: UUID(x), from_none], obj.get("parent_id"))
        aggregate_result = from_union([AggregateResult.from_dict, from_none], obj.get("aggregate_result"))
        attachments = from_union([lambda x: from_list(Attachment.from_dict, x), from_none], obj.get("attachments"))
        cite_metadata = from_union([CiteMetadata.from_dict, from_none], obj.get("_cite_metadata"))
        command = from_union([Command, from_none], obj.get("command"))
        args = from_union([lambda x: from_list(lambda x: from_union([from_int, from_str], x), x), from_none], obj.get("args"))
        status = from_union([MetadataStatus, from_none], obj.get("status"))
        citations = from_union([lambda x: from_list(Citation.from_dict, x), from_none], obj.get("citations"))
        return MessageMetadata(timestamp, message_type, finish_details, is_complete, model_slug, parent_id, aggregate_result, attachments, cite_metadata, command, args, status, citations)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.timestamp is not None:
            result["timestamp_"] = from_union([lambda x: to_enum(Timestamp, x), from_none], self.timestamp)
        if self.message_type is not None:
            result["message_type"] = from_none(self.message_type)
        if self.finish_details is not None:
            result["finish_details"] = from_union([lambda x: to_class(FinishDetails, x), from_none], self.finish_details)
        if self.is_complete is not None:
            result["is_complete"] = from_union([from_bool, from_none], self.is_complete)
        if self.model_slug is not None:
            result["model_slug"] = from_union([lambda x: to_enum(ModelSlug, x), from_none], self.model_slug)
        if self.parent_id is not None:
            result["parent_id"] = from_union([lambda x: str(x), from_none], self.parent_id)
        if self.aggregate_result is not None:
            result["aggregate_result"] = from_union([lambda x: to_class(AggregateResult, x), from_none], self.aggregate_result)
        if self.attachments is not None:
            result["attachments"] = from_union([lambda x: from_list(lambda x: to_class(Attachment, x), x), from_none], self.attachments)
        if self.cite_metadata is not None:
            result["_cite_metadata"] = from_union([lambda x: to_class(CiteMetadata, x), from_none], self.cite_metadata)
        if self.command is not None:
            result["command"] = from_union([lambda x: to_enum(Command, x), from_none], self.command)
        if self.args is not None:
            result["args"] = from_union([lambda x: from_list(lambda x: from_union([from_int, from_str], x), x), from_none], self.args)
        if self.status is not None:
            result["status"] = from_union([lambda x: to_enum(MetadataStatus, x), from_none], self.status)
        if self.citations is not None:
            result["citations"] = from_union([lambda x: from_list(lambda x: to_class(Citation, x), x), from_none], self.citations)
        return result


class MessageStatus(Enum):
    FINISHED_SUCCESSFULLY = "finished_successfully"
    IN_PROGRESS = "in_progress"


class MappingMessage:
    id: UUID
    author: Author
    create_time: Optional[float]
    update_time: Optional[float]
    content: MessageContent
    status: MessageStatus
    end_turn: Optional[bool]
    weight: float
    metadata: MessageMetadata
    recipient: Recipient

    def __init__(self, id: UUID, author: Author, create_time: Optional[float], update_time: Optional[float], content: MessageContent, status: MessageStatus, end_turn: Optional[bool], weight: float, metadata: MessageMetadata, recipient: Recipient) -> None:
        self.id = id
        self.author = author
        self.create_time = create_time
        self.update_time = update_time
        self.content = content
        self.status = status
        self.end_turn = end_turn
        self.weight = weight
        self.metadata = metadata
        self.recipient = recipient

    @staticmethod
    def from_dict(obj: Any) -> 'MappingMessage':
        assert isinstance(obj, dict)
        id = UUID(obj.get("id"))
        author = Author.from_dict(obj.get("author"))
        create_time = from_union([from_none, from_float], obj.get("create_time"))
        update_time = from_union([from_none, from_float], obj.get("update_time"))
        content = MessageContent.from_dict(obj.get("content"))
        status = MessageStatus(obj.get("status"))
        end_turn = from_union([from_bool, from_none], obj.get("end_turn"))
        weight = from_float(obj.get("weight"))
        metadata = MessageMetadata.from_dict(obj.get("metadata"))
        recipient = Recipient(obj.get("recipient"))
        return MappingMessage(id, author, create_time, update_time, content, status, end_turn, weight, metadata, recipient)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = str(self.id)
        result["author"] = to_class(Author, self.author)
        result["create_time"] = from_union([from_none, to_float], self.create_time)
        result["update_time"] = from_union([from_none, to_float], self.update_time)
        result["content"] = to_class(MessageContent, self.content)
        result["status"] = to_enum(MessageStatus, self.status)
        result["end_turn"] = from_union([from_bool, from_none], self.end_turn)
        result["weight"] = to_float(self.weight)
        result["metadata"] = to_class(MessageMetadata, self.metadata)
        result["recipient"] = to_enum(Recipient, self.recipient)
        return result


class Mapping:
    id: UUID
    message: Optional[MappingMessage]
    parent: Optional[UUID]
    children: List[UUID]

    def __init__(self, id: UUID, message: Optional[MappingMessage], parent: Optional[UUID], children: List[UUID]) -> None:
        self.id = id
        self.message = message
        self.parent = parent
        self.children = children

    @staticmethod
    def from_dict(obj: Any) -> 'Mapping':
        assert isinstance(obj, dict)
        id = UUID(obj.get("id"))
        message = from_union([MappingMessage.from_dict, from_none], obj.get("message"))
        parent = from_union([lambda x: UUID(x), from_none], obj.get("parent"))
        children = from_list(lambda x: UUID(x), obj.get("children"))
        return Mapping(id, message, parent, children)

    def to_dict(self) -> dict:
        result: dict = {}
        result["id"] = str(self.id)
        result["message"] = from_union([lambda x: to_class(MappingMessage, x), from_none], self.message)
        result["parent"] = from_union([lambda x: str(x), from_none], self.parent)
        result["children"] = from_list(lambda x: str(x), self.children)
        return result


class Conversation:
    title: Optional[str]
    create_time: float
    update_time: float
    mapping: Dict[str, Mapping]
    moderation_results: List[Any]
    current_node: UUID
    plugin_ids: None
    conversation_id: UUID
    conversation_template_id: None
    id: UUID

    def __init__(self, title: Optional[str], create_time: float, update_time: float, mapping: Dict[str, Mapping], moderation_results: List[Any], current_node: UUID, plugin_ids: None, conversation_id: UUID, conversation_template_id: None, id: UUID) -> None:
        self.title = title
        self.create_time = create_time
        self.update_time = update_time
        self.mapping = mapping
        self.moderation_results = moderation_results
        self.current_node = current_node
        self.plugin_ids = plugin_ids
        self.conversation_id = conversation_id
        self.conversation_template_id = conversation_template_id
        self.id = id

    @staticmethod
    def from_dict(obj: Any) -> 'Conversation':
        assert isinstance(obj, dict)
        title = from_union([from_none, from_str], obj.get("title"))
        create_time = from_float(obj.get("create_time"))
        update_time = from_float(obj.get("update_time"))
        mapping = from_dict(Mapping.from_dict, obj.get("mapping"))
        moderation_results = from_list(lambda x: x, obj.get("moderation_results"))
        current_node = UUID(obj.get("current_node"))
        plugin_ids = from_none(obj.get("plugin_ids"))
        conversation_id = UUID(obj.get("conversation_id"))
        conversation_template_id = from_none(obj.get("conversation_template_id"))
        id = UUID(obj.get("id"))
        return Conversation(title, create_time, update_time, mapping, moderation_results, current_node, plugin_ids, conversation_id, conversation_template_id, id)

    def to_dict(self) -> dict:
        result: dict = {}
        result["title"] = from_union([from_none, from_str], self.title)
        result["create_time"] = to_float(self.create_time)
        result["update_time"] = to_float(self.update_time)
        result["mapping"] = from_dict(lambda x: to_class(Mapping, x), self.mapping)
        result["moderation_results"] = from_list(lambda x: x, self.moderation_results)
        result["current_node"] = str(self.current_node)
        result["plugin_ids"] = from_none(self.plugin_ids)
        result["conversation_id"] = str(self.conversation_id)
        result["conversation_template_id"] = from_none(self.conversation_template_id)
        result["id"] = str(self.id)
        return result


def conversations_from_dict(s: Any) -> List[Conversation]:
    return from_list(Conversation.from_dict, s)


def conversations_to_dict(x: List[Conversation]) -> Any:
    return from_list(lambda x: to_class(Conversation, x), x)
