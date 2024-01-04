# Notice: For unknown reason the the author of this extension [@tkalayci71](https://github.com/tkalayci71) has deleted his GitHub account<br>This is a re-uploaded clone
original URL https://github.com/tkalayci71/embedding-inspector
Since the code is licensed under [Unlicense](LICENSE), I have re-uploaded the repository using a found fork on GitHub.

I take no credit and was not involved in development of this extension and I have no plans to maintaining it.

If someone wished to maintain this extension please get in contact.

Note:

Unfortunately I was not able to find the latest commit of this repository which has the hash

`448b6d06859557c446fa3f53df5e732a11608537` comment on `2023-03-18T12:01:18Z`

If someone has a copy of this please contact

---
---

# Embedding-inspector extension version 2.5 - 2022.12.08
for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions)

With this extension you can inspect internal/loaded embeddings and find out which embeddings are similar, and you can mix them to create new embeddings.

Inspired by [Embeddings editor](https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor.git) and [Tokenizer](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer.git) extensions.

# What's new

v2.0: SD2.0 and multi-vector support 

v2.1: Entering embedding ID instead of name is now supported, for example you can enter "#2368" instead of "cat"

v2.2: Entering a step value (like 1000) is now supported. This is needed only if you will continue training this embedding. Also, step and checkpoint info for loaded embeddings are now displayed.

v2.3: Added "List loaded embeddings" button

v2.4: Added "Concat mode" option. In this mode, embeddings will be just combined instead of being mixed. For example, "mona" and "lisa" can be combined into a single embedding "monalisa" which will contain 2 vectors, and the result will be the same as having "mona lisa" in the prompt, but with a single keyword.

v2.5 Added "global multiplier" option, which is useful to adjust strength in concat mode. Added a mini tokenizer. You can select "Send IDs to mixer" option to automate converting a short prompt to an embedding.

# Manual Installation

Download [embedding-inspector-main.zip](https://github.com/w-e-w/embedding-inspector/archive/refs/heads/main.zip) and extract into extensions folder.

# Usage

1) Enter a token name into "Text Input" box and click "Inspect" button. Only the first token found in the text input will be processed. Below, some information about the token will be displayed, and similar embeddings will be listed in the order of their similarity. This is useful to check if a word is in the token database, find internal tokens that are similar to loaded embeddings, and also to discover related unicode emojis.

![image](screenshot1.jpg)
![image](screenshot4.jpg)

2) Enter one or more token names in the "Name 0", "Name 1"... boxes, adjust their weights with "Multiplier" sliders, enter a unique name in "Filename" box, click "Save mixed" button. This will create a new embedding (mixed from the given embeddings and weights) and save it in the embeddings folder. If the file already exists, "Enable overwrite" box must be checked to allow overwriting it. Then, you use the filename as a keyword in your prompt.

![image](screenshot2.jpg)
![image](screenshot3.jpg)

3) Enter a short prompt in mini tokenizer text box, select "Send IDs to mixer" option, click "Tokenize". In the mixer section IDs will have been copied and "Concat mode" checked. Adjust multiplier and global multiplier sliders if necessary, enter a filename and click "Save mixed" button. Then use the filename as a keyword in your prompt.

![image](screenshot5.jpg)
![image](screenshot6.jpg)
![image](screenshot7.jpg)

# Background information

Stable Diffusion contains a database of ~49K words/tokens, and their numerical representations called embeddings. Your prompt is first tokenized using this database. For example, since the word "cat" is in the database it will be tokenized as a single item, but the word "catnip" is not in the database,  so will be tokenized as two items, "cat" and "nip". 

New tokens/concepts can also be loaded from embeddings folder. They are usually created via textual inversion, or you can download some from [Stable Diffusion concepts library](https://huggingface.co/sd-concepts-library). With Embedding-inspector you can inspect and mix embeddings both from the internal database and the loaded database.
