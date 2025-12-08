## Architecture Mapping
- Primary SDK code: `litellm/` (Python). Core entrypoints: `litellm/main.py` (chat/embeddings/moderation/audio), `litellm/router.py` (routing, fallbacks, deployments). Provider implementations live in `litellm/llms/`; shared utils in `litellm/litellm_core_utils/`.
- Supporting areas: `proxy/` (gateway server), `integrations/`, `types/`. Non-SDK: `tests/`, `docs/`, `ui/`.
- Scope: exported/public Python SDK functions; excludes tests/mocks/deprecated.

## Core Chat & Embedding APIs (`litellm/main.py`)
| Function Signature | Return Type | File/Module | Purpose Summary |
| --- | --- | --- | --- |
| `completion(model: str, messages: List, **kwargs)` | ModelResponse \| CustomStreamWrapper | litellm/main.py | Universal chat-completion entry point supporting tools, streaming, retries, and provider routing. |
| `completion_with_retries(*args, num_retries: int = 3, retry_strategy: Literal["constant_retry","exponential_backoff_retry"] = "constant_retry", original_function=completion, **kwargs)` | ModelResponse \| CustomStreamWrapper | litellm/main.py | Wraps `completion` with configurable retry logic using tenacity. |
| `acompletion_with_retries(*args, num_retries: int = 3, retry_strategy: str = "constant_retry", original_function=completion, **kwargs)` | Coroutine → ModelResponse \| CustomStreamWrapper | litellm/main.py | Async deprecated wrapper providing the same retry semantics as above. |
| `embedding(model, input, dimensions: Optional[int] = None, encoding_format: Optional[str] = None, *, aembedding: bool = False, **kwargs)` | EmbeddingResponse \| Coroutine[EmbeddingResponse] | litellm/main.py | Generates embeddings across providers; supports sync/async via `aembedding` flag. |
| `aembedding(*args, **kwargs)` | Coroutine → EmbeddingResponse | litellm/main.py | Async convenience wrapper delegating to `embedding`. |
| `text_completion(prompt: Union[str,List], model: Optional[str] = None, **kwargs)` | TextCompletionResponse | litellm/main.py | Legacy OpenAI-style text completion interface with provider translation. |
| `adapter_completion(adapter_id: str, **kwargs)` | BaseModel \| AdapterCompletionStreamWrapper \| None | litellm/main.py | Runs a completion via a registered adapter, translating input/output payloads. |

## Moderation & Audio (`litellm/main.py`)
| Function Signature | Return Type | File/Module | Purpose Summary |
| --- | --- | --- | --- |
| `moderation(input: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs)` | OpenAIModerationResponse | litellm/main.py | Synchronous moderation call (OpenAI-compatible) with LiteLLM response wrapping. |
| `amoderation(input: str, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs)` | Coroutine → OpenAIModerationResponse | litellm/main.py | Async moderation invocation with provider/key resolution. |
| `transcription(model: str, file: FileTypes, language: Optional[str] = None, prompt: Optional[str] = None, response_format: Optional[str] = None, **kwargs)` | TranscriptionResponse \| Coroutine[TranscriptionResponse] | litellm/main.py | Speech-to-text across OpenAI/Azure and other configured providers, with router-aware options. |
| `speech(model: str, input: str, voice: Optional[Union[str, dict]], **kwargs)` | HttpxBinaryResponseContent \| Coroutine[HttpxBinaryResponseContent] | litellm/main.py | Text-to-speech renderer mapping OpenAI-style params onto provider-specific configs. |

## Router Invocation APIs (`litellm/router.py`)
| Function Signature | Return Type | File/Module | Purpose Summary |
| --- | --- | --- | --- |
| `Router.__init__(model_list: Optional[List[DeploymentTypedDict]] = None, *, cache_responses: Optional[bool] = False, num_retries: Optional[int] = None, max_fallbacks: Optional[int] = None, routing_strategy: Literal["simple-shuffle","least-busy","usage-based","latency-based","mistral-router","openai"] = "simple-shuffle", **kwargs)` | Router | litellm/router.py | Configures routing pool, caching, retry/fallback policies, scheduling, and deployment metadata. |
| `Router.completion(model: str, messages: List[Dict[str,str]], **kwargs)` | ModelResponse \| CustomStreamWrapper | litellm/router.py | Synchronous chat completion with per-deployment selection and fallbacks. |
| `Router.acompletion(model: str, messages: List[AllMessageValues], stream: bool = False, **kwargs)` | Coroutine → ModelResponse \| CustomStreamWrapper | litellm/router.py | Async chat completion with scheduling and mid-stream fallback handling. |
| `Router.image_generation(prompt: str, model: str, **kwargs)` | Image generation response | litellm/router.py | Routes image-generation calls with retries/fallbacks. |
| `Router.aimage_generation(prompt: str, model: str, **kwargs)` | Coroutine → Image generation response | litellm/router.py | Async image-generation routing. |
| `Router.atranscription(file: FileTypes, model: str, **kwargs)` | Coroutine → TranscriptionResponse | litellm/router.py | Async speech-to-text routed across deployments. |
| `Router.aspeech(model: str, input: str, voice: str, **kwargs)` | Coroutine → HttpxBinaryResponseContent | litellm/router.py | Async text-to-speech via routed deployments. |
| `Router.text_completion(model: str, prompt: str, **kwargs)` | TextCompletionResponse | litellm/router.py | Routed legacy text completion call. |
| `Router.atext_completion(model: str, prompt: str, **kwargs)` | Coroutine → TextCompletionResponse | litellm/router.py | Async routed text completion. |
| `Router.aadapter_completion(adapter_id: str, model: str, **kwargs)` | Coroutine → BaseModel \| AdapterCompletionStreamWrapper | litellm/router.py | Async adapter-backed completion with deployment selection. |
| `Router.embedding(model: str, input: Union[str,List], **kwargs)` | EmbeddingResponse | litellm/router.py | Routed embedding generation with retries/fallbacks. |
| `Router.aembedding(model: str, input: Union[str,List], **kwargs)` | Coroutine → EmbeddingResponse | litellm/router.py | Async routed embeddings. |

## Router Deployment Management (`litellm/router.py`)
| Function Signature | Return Type | File/Module | Purpose Summary |
| --- | --- | --- | --- |
| `add_deployment(deployment: Deployment)` | Optional[Deployment] | litellm/router.py | Registers a new deployment, indexing by model id/name. |
| `upsert_deployment(deployment: Deployment)` | Optional[Deployment] | litellm/router.py | Adds or updates an existing deployment when params change. |
| `delete_deployment(id: str)` | Optional[Deployment] | litellm/router.py | Removes a deployment by id, updating indices. |
| `get_deployment(model_id: str)` | Optional[Deployment] | litellm/router.py | Fetches deployment details by id via O(1) index lookup. |
| `get_model_list(model_name: Optional[str] = None, team_id: Optional[str] = None)` | Optional[List[DeploymentTypedDict]] | litellm/router.py | Returns deployments (including wildcard and team-specific routes). |
| `get_model_ids(model_name: Optional[str] = None, exclude_team_models: bool = False)` | List[str] | litellm/router.py | Lists deployment ids, optionally filtered by model name or team inclusion. |
| `get_model_access_groups(model_name: Optional[str] = None, model_access_group: Optional[str] = None, team_id: Optional[str] = None)` | Dict[str, List[str]] | litellm/router.py | Resolves access group mappings for available routed models. |
