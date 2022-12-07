# Embedding-inspector extension version 2.1 - 2022.12.07
for ![AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions)

With this extension you can inspect internal/loaded embeddings and find out which embeddings are similar, and you can mix them to create new embeddings.

Inspired by [Embeddings editor](https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor.git) and ![Tokenizer](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer.git) extensions.

new in v2.1: entering embedding ID instead of name is now supported, for example you can enter "#2368" instead of "cat"

# Manual Installation

Download ![embedding-inspector-main.zip](https://github.com/tkalayci71/embedding-inspector/archive/refs/heads/main.zip) and extract into extensions folder.

# Usage

1) Enter a token name into "Text Input" box and click "Inspect" button. Only the first token found in the text input will be processed. Below, some information about the token will be displayed, and similar embeddings will be listed in the order of their similarity. This is useful to check if a word is in the token database, find internal tokens that are similar to loaded embeddings, and also to discover related unicode emojis.

![image](screenshot1.jpg)
![image](screenshot4.jpg)

2) Enter one or more token names in the "Name 0", "Name 1"... boxes, adjust their weights with "Multiplier" sliders, enter a unique name in "Filename" box, click "Save mixed" button. This will create a new embedding (mixed from the given embeddings and weights) and save it in the embeddings folder. If the file already exists, "Enable overwrite" box must be checked to allow overwriting it.  Then, you use the filename as a keyword in your prompt.

![image](screenshot2.jpg)
![image](screenshot3.jpg)

# Background information

Stable Diffusion contains a database of ~49K words/tokens, and their numerical representations called embeddings. Your prompt is first tokenized using this database. For example, since the word "cat" is in the database it will be tokenized as a single item, but the word "catnip" is not in the database,  so will be tokenized as two items, "cat" and "nip". 

New tokens/concepts can also be loaded from embeddings folder. They are usually created via textual inversion, or you can download some from [Stable Diffusion concepts library](https://huggingface.co/sd-concepts-library). With Embedding-inspector you can inspect and mix embeddings both from the internal database and the loaded database.
