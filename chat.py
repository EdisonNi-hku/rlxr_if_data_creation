import openai
import os
import sys
import time
import logging
import json
import diskcache as dc
import threading
import requests
from openai import OpenAI

log = logging.getLogger()


# Reuse the same generation configuration style as in step_fact_check_thinking
GENERATION_CONFIGS = {
    'openai/gpt-oss-120b': {
        'extra_body': {"reasoning_effort": 'medium'},
    },
    "Qwen/Qwen3-8B": {
        "temperature": 0.6,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 20},
    },
    "microsoft/Phi-4-reasoning-plus": {
        "temperature": 0.8,
        "top_p": 0.95,
        "extra_body": {"enable_thinking": True, "top_k": 50},
    },
    "gpt-5.1-2025-11-13": {"reasoning_effort": 'medium'},
    "gpt-5-mini-2025-08-07": {"reasoning_effort": 'medium'},
}


class LocalChat:
    """
    Local OpenAI-compatible chat client with persistent disk cache.
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        base_url: str = "http://localhost:8000/v1",
        cache_path: str = os.path.expanduser("~") + "/.cache",
        generation_config: dict = None,
        cache_is_file: bool = False,
    ):
        if cache_is_file:
            self.cache_path = cache_path
            cache_dir = os.path.dirname(cache_path) or "."
        else:
            cache_dir = cache_path
            self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")

        os.makedirs(cache_dir, exist_ok=True)

        self.base_url = base_url
        self.model = model
        self.generation_config = GENERATION_CONFIGS[model] if generation_config is None else generation_config
        self.client = OpenAI(base_url=self.base_url, api_key='EMPTY')

        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()

    def ask(self, messages: list[dict], **kwargs) -> str:
        cache_key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        reply, reasoning_content = self.cache.get((self.model, cache_key), ('', ''))
        if reply == '':
            chat = self._send_request(messages, self.generation_config)
            if chat is None:
                reply, reasoning_content = '', ''
            else:
                reasoning_content = getattr(chat.choices[0].message, "reasoning_content", "")
                reply = chat.choices[0].message.content
                if '</think>' in reply:
                    response_list = reply.split('</think>')
                    reply = response_list[1].strip()
                    reasoning_content = response_list[0].strip()

            with self._lock:
                self.cache[(self.model, cache_key)] = (reply, reasoning_content)
        else:
            # pass
            print("Loaded from cache")

        if not reply:
            print("Reply is empty")
            print(f"Chat object: {chat}")

        low = reply.lower()
        if "please provide" in low or "to assist you" in low or "as an ai language model" in low:
            return "", ""

        return reply, reasoning_content

    def _send_request(self, messages, generation_config):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **generation_config,
            )
        except Exception as e:
            log.info(f"Request to OpenAI failed with exception: {e}.")
            return None

        return response


class DeepSeekChat:
    def __init__(
            self,
            cache_path: str,
            api_base: str | None = "https://api.deepseek.com/v1",
            model: str = "deepseek-reasoner",
            api_key: str | None = None,
            wait_times: tuple = (5, 10),
    ):
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        if cache_path is None:
            cache_path = '~/.cache'
        self.cache_path = os.path.join(cache_path, "deepseek_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.wait_times = wait_times
        
        # Initialize cache with proper settings
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()
        print(f"Using {self.api_base}")
        self.client = openai.OpenAI(base_url=self.api_base, api_key=self.api_key)

    def ask(self, message: str, json_output=False) -> str:
        # First try to get from cache without lock
        reply = self.cache.get((self.model, message), '')
        
        if reply == '':
            if self.api_key is None:
                raise Exception("Cant ask DeepSeek without token.")
            messages = [
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages, json_output)
            if chat is None:
                reply = ""
            else:
                reply = chat.choices[0].message.content
            # Only lock when writing to cache
            with self._lock:
                self.cache[(self.model, message)] = reply

        if any(x in reply.lower() for x in ["please provide", "to assist you", "as an ai language model"]):
            return ""

        return reply

    def _send_request(self, messages, json_output=False):
        chat_args = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.6,
        }
        if json_output:
            chat_args['response_format'] = {'type': 'json_object'}
        for i in range(len(self.wait_times)):
            try:
                return self.client.chat.completions.create(**chat_args)
            except Exception as e:
                sleep_time = self.wait_times[i]
                log.info(
                    f"Request failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)
        try:
            return self.client.chat.completions.create(**chat_args)
        except Exception as e:
            sys.stderr.write(f'Error: {e}')
            return None

    def __del__(self):
        """Cleanup method to properly close the cache when the instance is destroyed."""
        if hasattr(self, 'cache'):
            self.cache.close()


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        base_url: str = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        generation_config: dict = {},
        system_prompt: str = "You are an intelligent assistant.",
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.base_url = base_url
        self.generation_config = GENERATION_CONFIGS[openai_model] if generation_config is None else generation_config
        self.system_prompt = system_prompt

    def ask(self, message: str, json_output=False) -> str:
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        openai_responses = dc.Cache(self.cache_path, **cache_settings)

        if (self.openai_model, message) in openai_responses:
            reply = openai_responses[(self.openai_model, message)]

        else:
            # Ask openai
            if openai.api_key is None:
                raise Exception(
                    "Cant ask openAI without token. "
                    "Please specify OPENAI_API_KEY in environment parameters."
                )
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ]
            else:
                messages = [
                    {"role": "user", "content": message},
                ]
            chat = self._send_request(messages)
            reply = chat.choices[0].message.content

            openai_responses[(self.openai_model, message)] = reply
            openai_responses.close()

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        for i in range(len(sleep_time_values)):
            try:
                return openai.OpenAI(base_url=self.base_url).chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    **self.generation_config,
                )
            except Exception as e:
                sleep_time = sleep_time_values[i]
                log.info(
                    f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return openai.OpenAI(base_url=self.base_url).chat.completions.create(
            model=self.openai_model,
            messages=messages,
            **self.generation_config,
        )


class ApiChat:
    """
    Generic LLM API client with persistent disk cache.

    Wraps a POST-based LLM endpoint that expects:
        {model, prompt, params, app, quota_id, user_id, access_key, tag}

    Config is loaded from a JSON file (default: config/llm_api.json).
    Interface matches LocalChat.ask() so it can be used as a drop-in replacement.
    """

    def __init__(
        self,
        config_path: str = "config/llm_api.json",
        model: str = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        generation_config: dict = None,
        debug: bool = False,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.url = self.config["url"]
        self.model = model or self.config["model"]
        self.app = self.config.get("app", "")
        self.quota_id = self.config.get("quota_id", "")
        self.user_id = self.config.get("user_id", "")
        self.access_key = self.config.get("access_key", "")
        self.tag = self.config.get("tag", "")
        self.debug = debug
        self._debug_first_call = True  # print full details for the first API call

        # Params: explicit generation_config overrides config file defaults
        if generation_config is not None:
            self.params = generation_config
        else:
            self.params = self.config.get("default_params", {})

        print(f"[ApiChat] url={self.url}")
        print(f"[ApiChat] model={self.model}")
        print(f"[ApiChat] params={json.dumps(self.params)}")
        print(f"[ApiChat] app={self.app}  quota_id={self.quota_id}  "
              f"user_id={self.user_id}  tag={self.tag}")

        # Cache setup (same pattern as LocalChat)
        self.cache_path = os.path.join(cache_path, "api_chat_cache.diskcache")
        os.makedirs(cache_path, exist_ok=True)

        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()
        self._session = requests.Session()

    def ask(self, messages: list[dict], **kwargs) -> tuple[str, str]:
        """Send messages to the API and return (reply, reasoning_content).

        Messages can be in either format:
          - OpenAI style: [{"role": "user", "content": "text"}]
          - Structured:   [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
        Both are accepted; simple string content is auto-wrapped for the API.
        """
        params = {**self.params, **kwargs} if kwargs else self.params
        cache_key = json.dumps(
            {"messages": messages, "params": params},
            ensure_ascii=False, sort_keys=True,
        )
        reply, reasoning_content = self.cache.get(
            (self.model, cache_key), ("", "")
        )
        if reply != "":
            return reply, reasoning_content

        # Normalize messages: wrap plain-string content into structured format
        prompt = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            prompt.append({"role": msg["role"], "content": content})

        response_json = self._send_request(prompt, params)
        if response_json is None:
            print("[ApiChat] _send_request returned None")
            reply, reasoning_content = "", ""
        else:
            reply = self._extract_reply(response_json)
            reasoning_content = self._extract_reasoning(response_json)
            if not reply:
                # Reply extraction failed — dump the response for debugging
                resp_str = json.dumps(response_json, ensure_ascii=False)
                print(f"[ApiChat] _extract_reply returned empty string")
                print(f"[ApiChat] Response keys: {list(response_json.keys()) if isinstance(response_json, dict) else type(response_json).__name__}")
                print(f"[ApiChat] Full response (first 1000 chars): {resp_str[:1000]}")
            if "</think>" in reply:
                parts = reply.split("</think>")
                reply = parts[1].strip()
                reasoning_content = parts[0].strip()

        with self._lock:
            self.cache[(self.model, cache_key)] = (reply, reasoning_content)

        low = reply.lower()
        if "please provide" in low or "to assist you" in low or "as an ai language model" in low:
            return "", ""

        return reply, reasoning_content

    def _send_request(self, prompt: list[dict], params: dict) -> dict | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "params": params,
            "app": self.app,
            "quota_id": self.quota_id,
            "user_id": self.user_id,
            "access_key": self.access_key,
            "tag": self.tag,
        }

        # Print the full payload for the very first API call
        if self._debug_first_call:
            self._debug_first_call = False
            payload_debug = dict(payload)
            # Truncate prompt content for readability
            payload_debug["prompt"] = f"[{len(prompt)} messages, first role={prompt[0]['role']}]"
            print(f"[ApiChat] First request payload (without prompt body):")
            print(f"  POST {self.url}")
            print(f"  {json.dumps(payload_debug, ensure_ascii=False, indent=2)}")

        try:
            resp = self._session.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            if resp.status_code != 200:
                print(f"[ApiChat] HTTP {resp.status_code} from {self.url}")
                print(f"[ApiChat] Response body: {resp.text[:2000]}")
            resp.raise_for_status()
            result = resp.json()
            if self.debug:
                print(f"[ApiChat DEBUG] Full response:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
        except Exception as e:
            print(f"[ApiChat] Request failed: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _extract_reply(data: dict) -> str:
        """Extract the reply text from various possible response formats.

        Priority order:
          1. OpenAI choices[0].message.content (at current level)
          2. Recurse into "completion" / "data" sub-dicts
          3. Flat string keys as last resort
        """
        # OpenAI-compatible: choices[0].message.content
        if "choices" in data:
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        # Recurse into nested structures before trying flat keys,
        # because top-level "message" can be "success" (not the LLM reply).
        for nest_key in ("completion", "data"):
            if nest_key in data and isinstance(data[nest_key], dict):
                result = ApiChat._extract_reply(data[nest_key])
                if result:
                    return result
        # Flat fields (last resort)
        for key in ("response", "content", "text", "reply", "output", "message"):
            if key in data and isinstance(data[key], str):
                return data[key]
        return ""

    @staticmethod
    def _extract_reasoning(data: dict) -> str:
        """Extract reasoning/thinking content if present."""
        for key in ("reasoning_content", "thinking", "reasoning"):
            if key in data and isinstance(data[key], str):
                return data[key]
        if "choices" in data:
            try:
                msg = data["choices"][0]["message"]
                for key in ("reasoning_content", "thinking"):
                    if key in msg and isinstance(msg[key], str):
                        return msg[key]
            except (KeyError, IndexError, TypeError):
                pass
        # Recurse into nested structures
        for nest_key in ("completion", "data"):
            if nest_key in data and isinstance(data[nest_key], dict):
                result = ApiChat._extract_reasoning(data[nest_key])
                if result:
                    return result
        return ""