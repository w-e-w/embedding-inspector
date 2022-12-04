# Embedding Inspector extension for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/tkalayci71/embedding-inspector
# version 1.0 - 2022.12.04
#
# TODO: support for SD 2.0

import gradio as gr
from modules import script_callbacks, shared, sd_hijack
import torch, os

MAX_NUM_MIX = 5
MAX_SIMILAR_EMBS = 30

#-------------------------------------------------------------------------------

def get_embedding_info(text):

    try:
        tokenizer = shared.sd_model.cond_stage_model.wrapped.tokenizer

        text_model = shared.sd_model.cond_stage_model.wrapped.transformer.text_model
        internal_embs = text_model.embeddings.token_embedding.wrapped.weight
        loaded_embs = sd_hijack.model_hijack.embedding_db.word_embeddings

        loaded_emb = loaded_embs.get(text, None)
        if loaded_emb!=None:
            emb_name = loaded_emb.name
            emb_id = '['+loaded_emb.checksum()+']'
            emb_vec = loaded_emb.vec
            return emb_name, emb_id, emb_vec

        emb_id = (tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])[0]

        emb_name_utf8 = tokenizer.decoder.get(emb_id)
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')

        emb_vec = internal_embs[emb_id].unsqueeze(0)

        return emb_name, emb_id, emb_vec
    except:
        return None, None, None

#-------------------------------------------------------------------------------

def do_inspect(text):

    results = []

    try:

        tokenizer = shared.sd_model.cond_stage_model.wrapped.tokenizer
        text_model = shared.sd_model.cond_stage_model.wrapped.transformer.text_model
        internal_embs = text_model.embeddings.token_embedding.wrapped.weight
        loaded_embs = sd_hijack.model_hijack.embedding_db.word_embeddings

        emb_name, emb_id, emb_vec = get_embedding_info(text)
        if (emb_name==None) or (emb_id==None) or (emb_vec==None):
            results.append('An error occurred')
            return

        results.append('Name: "'+emb_name+'"')
        results.append('ID: '+str(emb_id))
        results.append('Vector: shape: '+str(emb_vec.shape)+', dtype: '+str(emb_vec.dtype)+', device: '+str(emb_vec.device))
        results.append('')

        for v in range(emb_vec.shape[0]):

            results.append('--- Vector #'+str(v))
            input1 = internal_embs.to(device='cpu',dtype=torch.float32)
            input2 = emb_vec[v].to(device='cpu',dtype=torch.float32)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            scores = cos(input1, input2)
            sorted_scores, sorted_ids = torch.sort(scores, descending=True)
            best_ids = sorted_ids[0:MAX_SIMILAR_EMBS].numpy()

            results.append("Similar embeddings:")
            r = []
            for i in range(0, MAX_SIMILAR_EMBS):
                emb_id = best_ids[i].item()
                emb_name_utf8 = tokenizer.decoder.get(emb_id)
                byte_array = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
                emb_name = byte_array.decode("utf-8", errors='backslashreplace').strip()
                r.append(emb_name+'('+str(emb_id)+')')
            results.append('   '.join(r))
            results.append('')

    finally:
        return '\n'.join(results)

#-------------------------------------------------------------------------------

def do_save(*args):

    results = []

    try:
        save_name = args[-2].strip()
        enable_overwrite = args[-1]
        if save_name=='':
            results.append('Filename empty')
            return

        save_filename = 'embeddings/'+save_name+'.bin'
        file_exists = os.path.exists(save_filename)
        if file_exists:
            results.append('File already exists')
            if not(enable_overwrite):
                results.append('Aborting')
                return

        tot_vec = torch.zeros(768)
        for k in range(MAX_NUM_MIX):
            name= args[k]
            mixval = args[k+MAX_NUM_MIX]
            if (name=='') or (mixval==0): continue
            emb_name, emb_id, emb_vec = get_embedding_info(name)
            mix_vec = emb_vec[0].to(device='cpu',dtype=torch.float32)
            mix_vec *= mixval
            tot_vec += mix_vec
            results.append('+ '+emb_name+'('+str(emb_id)+')'+' x '+str(mixval))

        from modules.textual_inversion.textual_inversion import Embedding
        tot_vec = tot_vec.unsqueeze(0)
        new_emb = Embedding(tot_vec, save_name)
        try:
            new_emb.save(save_filename)
            results.append('Saved "'+save_filename+'"')
        except:
            results.append('Error saving "'+save_filename+'"')

        sd_hijack.model_hijack.embedding_db.dir_mtime=0
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        results.append('Reloading embeddings')

    finally:
        return '\n'.join(results)

#-------------------------------------------------------------------------------

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            with gr.Row():

                with gr.Column(variant='panel'):
                    text_input = gr.Textbox(label="Text input", lines=1)
                    inspect_button = gr.Button(value="Inspect", variant="primary")
                    inspect_result = gr.Textbox(label="Results", lines=15)

                with gr.Column(variant='panel'):
                    mix_inputs = []
                    mix_sliders = []
                    for n in range(MAX_NUM_MIX):
                        with gr.Row():
                           with gr.Column():
                               mix_inputs.append(gr.Textbox(label="Mix"+str(n)+" name", lines=1))
                           with gr.Column():
                               mix_sliders.append(gr.Slider(label="Mul",value=0.5,minimum=-1.00, maximum=1.00, step=0.1))
                    with gr.Row():
                        save_name = gr.Textbox(label="filename",lines=1)
                        save_button = gr.Button(value="Save mixed", variant="primary")
                        enable_overwrite = gr.Checkbox(value=False,label="Enable overwrite")

                    with gr.Row():
                        save_result = gr.Textbox(label="Log", lines=5)

            inspect_button.click(fn=do_inspect,inputs=[text_input],outputs=[inspect_result])
            save_button.click(fn=do_save, inputs=mix_inputs+mix_sliders+[save_name,enable_overwrite],outputs=save_result)

    return [(ui, "Embedding Inspector", "inspector")]

script_callbacks.on_ui_tabs(add_tab)
