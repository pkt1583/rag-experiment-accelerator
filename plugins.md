# rag-exp-acc plugins


## Why

The main reason behind this proposal is to allow users to write their own classes that can be used within the rag-experiment-accelerator. This would make the tool more modular and give a way for users to implement their own code without having to modify the source code. In addition, community plugins could be written for reuse if it is not integrated or supported by the tool.


For example, A user could decide that they want to use an LLM model that we do not support, like OpenAI models. They could then write their own plugin that gets invoked by the rag-exp-acc tool.

> note: The examples below show how this can be done with chat models, but this also could be expanded to other aspects of the tool like embedding models, preprocessors, chunking, doc loading, reranking, searching, etc. 


## Examples

### OpenAI chat model

A user could write the `Plugin` class by implmenting the methods of `ChatModel`.

```python
# rag_experiment_accelerator/chat/chat_model.py
class ChatModel(ABC):

    @abstractmethod
    def chat(user_prompt: str, system_prompt: str) -> str:
        pass
```

```python
class Plugin(ChatModel):
    def __init__(self, model_name: str, temperature: float, api_key: str):
        self._deployment_name = deployment_name
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key)

    def chat(self, user_prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self._client.chat.completions.create(
            model=self._deployment_name,
            messages=messages,
            temperature=self._temperature,
        )
        return response.choices[0].message.content
```


## Components


### config.json

```json
    "chat_model": {
        // "type" and "module" are common across plugins
        "type": "plugin", // must be 'plugin' if it is a plugin
        "module": "path.to.module",

        // everything below here is specific to a custom plugin. It represents the argument names and values that the plugin accepts for initialization
        "model_name": "gpt-35-turbo",
        "temperature": 0,
        // from_env would be a reserved key as it provides a way to specify values coming .env
        "from_env": {
            "openai_api_key": { // this is the argument name
                "env_name": "OPENAI_API_KEY", // the value is the value of this environment var
                "mask": true, // defualts to true - we likely dont want to print env variables 
                "critical": false // defualt is up for debate here
            },
            "openai_api_version": {
                "env_name": "OPENAI_API_VERSION"
            },
            "openai_endpoint": {
                "env_name": "OPENAI_ENDPOINT"
            }
        }
    }
```

This example demonstrates a chat model for the whole application (this is how it currently works). A nice feature might be for each script to accept its own chat model. This would allow a user to use different models and parameters depending on the use case. For instance, A user may not always want to use the same model for qa generation as they do for summarising a piece of text for upload to Azure AI search, or they may want to set different temperature values for the same model.


### Factory

We need a way to load the plugins. Using a factory could work well in this case.

```python
class ChatModelFactory:
    @staticmethod
    def create(type: str, **kwargs) -> T:
        if type == "azure":
            # AOAIChatModel is a supported model and is in the rag-exp-acc source code. 
            # However, this could be written to just be a plugin. 
            # If we made each plugable component a plugin, we could resuse this code across plugins,
            # and not just for specific plugin type, ex: chat vs embedding model plugins
            return AOAIChatModel(**kwargs)
        elif type == "plugin": # these are for custom plugins
            # get the name of the module
            module_name = kwargs.pop("module", None)

            # error handling
            if not module_name:
                raise ValueError(
                    "Invalid embedding type: plugin. Must provide a module name."
                )

            # import the module
            module = importlib.import_module(module_name)

            # return an instance of the plugin class using the arguments from config
            # note: class must be class Plugin
            return module.Plugin(**kwargs)
        else:
            raise ValueError(
                f"Invalid embedding type: {type}. Must be one of ['azure', 'plugin']"
            )
```

### config.py

Each plugin should be loaded into the config so that it can be used within the tool. This way, the scripts can use the plugins directly by just calling the method it needs. 

Ex: 
```python
config.chat_model.chat("my user prompt", "my system prompt")
```


An example of how a chat model can be loaded from config.json

```python
# ...code from rag_experiment_accelerator/config/config.py in Config._initialize
    chat_model_config = data.get("chat_model")
    chat_model_kwargs = self._handle_get_plugin_env_vars(chat_model_config)
    self.chat_model: ChatModel = ChatModelFactory.create(**chat_model_kwargs)
# ...more code


    # if we have a key of 'from_env', we take the name of each key and get its value from os.env
    # we set the key name and value on a copy of the config dict
    # before returning, 'from_env' is deleted from the copied dict
    def _handle_get_plugin_env_vars(self, plugin_config: dict):
        # copy dict
        return_config = dict(plugin_config)
        env_vars = return_config.get("from_env")
        if env_vars:
            # all "from_env" key-values must be a dict
            if not isinstance(env_vars, dict):
                raise ValueError("from_env must be a dictionary")
            
            for k, v in env_vars.items():
                if not isinstance(v, dict):
                    raise ValueError("from_env values must be a dictionary")
                # set the key on the dict with the value from os.env
                return_config[k] = _get_env_var(v["env_name"], v.get("critical", False), v.get("mask", True))

            # delete "from_env" because we have the arg name and values in the dict
            del return_config["from_env"]

        return return_config

```


## Requirments and limitations
- The class must implment the methods of a known abstract base class.
- the config object must include the path to the module
- the plugin must have a consistent name like `Plugin` to enable creating an instance of the class from the module
- a plugin is limited to accepting built in data types. Ex: a plugin couldn't accept `OpenAICredentials`. It would need to take key-value pairs, a list, or dict. These values could then be used within the plugin itself to create any needed python classes.
