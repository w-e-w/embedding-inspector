# Embedding Inspector extension for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/tkalayci71/embedding-inspector
# version 2.0 - 2022.12.06
#

import gradio as gr
from modules import script_callbacks, shared, sd_hijack
import torch, os
from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
from modules.textual_inversion.textual_inversion import Embedding

MAX_NUM_MIX = 5 #number of embeddings that can be mixed
MAX_SIMILAR_EMBS = 30 #number of similar embeddings to show
VEC_SHOW_TRESHOLD = 1 #formatting for printing tensors
SEP_STR = '-'*80 #seperator string

#-------------------------------------------------------------------------------

def get_data():

    loaded_embs = sd_hijack.model_hijack.embedding_db.word_embeddings

    embedder = shared.sd_model.cond_stage_model.wrapped
    if embedder.__class__.__name__=='FrozenCLIPEmbedder': #SD1.x detected
        tokenizer = embedder.tokenizer
        internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

    elif embedder.__class__.__name__=='FrozenOpenCLIPEmbedder': #SD2.0 detected
        tokenizer = open_clip_tokenizer
        internal_embs = embedder.model.token_embedding.wrapped.weight

    else:
        tokenizer = None
        internal_embs = None

    return tokenizer, internal_embs, loaded_embs #return these useful references

#-------------------------------------------------------------------------------

def text_to_emb_ids(text, tokenizer):

    if tokenizer.__class__.__name__== 'CLIPTokenizer': #SD1.x detected
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    elif tokenizer.__class__.__name__== 'SimpleTokenizer': #SD2.0 detected
        emb_ids =  tokenizer.encode(text)

    else:
        emb_ids = None

    return emb_ids #return list of embedding IDs for text

#-------------------------------------------------------------------------------

def emb_id_to_name(emb_id, tokenizer):

    emb_name_utf8 = tokenizer.decoder.get(emb_id)

    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else:
        emb_name = '!Unknown ID!'

    return emb_name #return embedding name for embedding ID

#-------------------------------------------------------------------------------

def get_embedding_info(text):

    tokenizer, internal_embs, loaded_embs = get_data()

    loaded_emb = loaded_embs.get(text, None)
    if loaded_emb!=None:
        emb_name = loaded_emb.name
        emb_id = '['+loaded_emb.checksum()+']' #string for loaded embeddings
        emb_vec = loaded_emb.vec
        return emb_name, emb_id, emb_vec

    emb_ids = text_to_emb_ids(text, tokenizer)
    if len(emb_ids)==0: return None, None, None

    emb_id = emb_ids[0] #int for internal embeddings
    emb_name = emb_id_to_name(emb_id, tokenizer)
    emb_vec = internal_embs[emb_id].unsqueeze(0)

    return emb_name, emb_id, emb_vec #return embedding name, ID, vector for first token in text

#-------------------------------------------------------------------------------

def do_inspect(text):

    #get the embedding info for first token in text
    emb_name, emb_id, emb_vec = get_embedding_info(text)
    if (emb_name==None) or (emb_id==None) or (emb_vec==None):
        return 'An error occurred'

    results = []

    #add embedding info to results
    results.append('Embedding name: "'+emb_name+'"')

    if type(emb_id)==int:
        results.append('Embedding ID: '+str(emb_id)+' (internal)')
    else:
        results.append('Embedding ID: '+str(emb_id)+' (loaded)')

    vec_count = emb_vec.shape[0]
    vec_size = emb_vec.shape[1]
    results.append('Vector count: '+str(vec_count))
    results.append('Vector size: '+str(vec_size))
    results.append(SEP_STR)

    #add vector infos to results
    tokenizer, internal_embs, loaded_embs = get_data()
    input1 = None #this will contain all internal embeddings copied to cpu as float32
    for v in range(vec_count):

        torch.set_printoptions(threshold=VEC_SHOW_TRESHOLD,profile='default')
        results.append('Vector['+str(v)+'] = '+str(emb_vec[v]))

        #calculate similar embeddings and add to results
        if input1==None:
            if vec_size==internal_embs.shape[1]:
                input1 = internal_embs.to(device='cpu',dtype=torch.float32)
            else:
                results.append('Vector size is not compatible with current SD model')

        if input1!=None:
            results.append('')
            results.append("Similar embeddings:")
            input2 = emb_vec[v].to(device='cpu',dtype=torch.float32)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            scores = cos(input1, input2)
            sorted_scores, sorted_ids = torch.sort(scores, descending=True)
            best_ids = sorted_ids[0:MAX_SIMILAR_EMBS].numpy()
            r = []
            for i in range(0, MAX_SIMILAR_EMBS):
                emb_id = best_ids[i].item()
                emb_name = emb_id_to_name(emb_id, tokenizer)
                r.append(emb_name+'('+str(emb_id)+')')
            results.append('   '.join(r))

        results.append(SEP_STR)

    return '\n'.join(results) #return info string to results textbox

#-------------------------------------------------------------------------------

def do_save(*args):

    results = []

    #do some checks
    save_name = args[-2].strip()
    enable_overwrite = args[-1]
    if save_name=='':return('Filename is empty')

    save_filename = 'embeddings/'+save_name+'.bin'
    file_exists = os.path.exists(save_filename)
    if (file_exists):
        if not(enable_overwrite):
            return('File already exists, overwrite not enabled, aborting save.')
        else:
            results.append('File already exists, overwrite is enabled')


    #calculate mixed embedding in tot_vec
    vec_size = None
    tot_vec = None
    for k in range(MAX_NUM_MIX):
        name= args[k]
        mixval = args[k+MAX_NUM_MIX]
        if (name=='') or (mixval==0): continue

        emb_name, emb_id, emb_vec = get_embedding_info(name)
        mix_vec = emb_vec.to(device='cpu',dtype=torch.float32)

        if vec_size==None:
            vec_size = mix_vec.shape[1]
        else:
            if vec_size!=mix_vec.shape[1]:
                results.append('! Vector size is not compatible, skipping '+emb_name+'('+str(emb_id)+')')
                continue

        if tot_vec==None:
            tot_vec = torch.zeros(vec_size).unsqueeze(0)

        if mix_vec.shape[0]!=tot_vec.shape[0]:
            padding = torch.zeros(abs(tot_vec.shape[0]-mix_vec.shape[0]),vec_size)
            if mix_vec.shape[0]<tot_vec.shape[0]:
                mix_vec = torch.cat([mix_vec, padding])
            else:
                tot_vec = torch.cat([tot_vec, padding])

        tot_vec+= mix_vec * mixval
        results.append('+ '+emb_name+'('+str(emb_id)+')'+' x '+str(mixval))

    #save the mixed embedding
    if (tot_vec==None):
        results.append('No embeddings were mixed, nothing to save')
    else:
        new_emb = Embedding(tot_vec, save_name)
        try:
            new_emb.save(save_filename)
            results.append('Saved "'+save_filename+'"')
        except:
            results.append('Error saving "'+save_filename+'"')

        results.append('Reloading all embeddings')
        sd_hijack.model_hijack.embedding_db.dir_mtime=0
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return '\n'.join(results)  #return info string to log textbox

#-------------------------------------------------------------------------------

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            with gr.Row():

                with gr.Column(variant='panel'):
                    text_input = gr.Textbox(label="Text input", lines=1, placeholder="Enter embedding name (only first token will be processed)")
                    inspect_button = gr.Button(value="Inspect", variant="primary")
                    inspect_result = gr.Textbox(label="Results", lines=15)

                with gr.Column(variant='panel'):
                    mix_inputs = []
                    mix_sliders = []
                    for n in range(MAX_NUM_MIX):
                        with gr.Row():
                           with gr.Column():
                               mix_inputs.append(gr.Textbox(label="Name "+str(n), lines=1, placeholder="Enter name of embedding to mix"))
                           with gr.Column():
                               mix_sliders.append(gr.Slider(label="Multiplier",value=0.5,minimum=-1.00, maximum=1.00, step=0.1))
                    with gr.Row():
                        save_name = gr.Textbox(label="Filename",lines=1,placeholder='Enter file name to save')
                        save_button = gr.Button(value="Save mixed", variant="primary")
                        enable_overwrite = gr.Checkbox(value=False,label="Enable overwrite")

                    with gr.Row():
                        save_result = gr.Textbox(label="Log", lines=5)

            inspect_button.click(fn=do_inspect,inputs=[text_input],outputs=[inspect_result])
            save_button.click(fn=do_save, inputs=mix_inputs+mix_sliders+[save_name,enable_overwrite],outputs=save_result)

    return [(ui, "Embedding Inspector", "inspector")]

script_callbacks.on_ui_tabs(add_tab)
