# Embedding Inspector extension for AUTOMATIC1111/stable-diffusion-webui
#
# https://github.com/tkalayci71/embedding-inspector
# version 2.54 - 2022.12.29
#

import gradio as gr
from modules import script_callbacks, shared, sd_hijack
import torch, os
from modules.textual_inversion.textual_inversion import Embedding
import math, random

MAX_NUM_MIX = 6 # number of embeddings that can be mixed
MAX_SIMILAR_EMBS = 30 # number of similar embeddings to show
VEC_SHOW_TRESHOLD = 1 # change to 10000 to see all values
VEC_SHOW_PROFILE = 'default' #change to 'full' for more precision
SEP_STR = '-'*80 # separator string

ENABLE_GRAPH = False
ENABLE_SHOW_CHECKSUM = False #slows down listing loaded embeddings

#-------------------------------------------------------------------------------

def get_data():

    loaded_embs = sd_hijack.model_hijack.embedding_db.word_embeddings

    embedder = shared.sd_model.cond_stage_model.wrapped
    if embedder.__class__.__name__=='FrozenCLIPEmbedder': # SD1.x detected
        tokenizer = embedder.tokenizer
        internal_embs = embedder.transformer.text_model.embeddings.token_embedding.wrapped.weight

    elif embedder.__class__.__name__=='FrozenOpenCLIPEmbedder': # SD2.0 detected
        from modules.sd_hijack_open_clip import tokenizer as open_clip_tokenizer
        tokenizer = open_clip_tokenizer
        internal_embs = embedder.model.token_embedding.wrapped.weight

    else:
        tokenizer = None
        internal_embs = None

    return tokenizer, internal_embs, loaded_embs # return these useful references

#-------------------------------------------------------------------------------

def text_to_emb_ids(text, tokenizer):

    text = text.lower()

    if tokenizer.__class__.__name__== 'CLIPTokenizer': # SD1.x detected
        emb_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]

    elif tokenizer.__class__.__name__== 'SimpleTokenizer': # SD2.0 detected
        emb_ids =  tokenizer.encode(text)

    else:
        emb_ids = None

    return emb_ids # return list of embedding IDs for text

#-------------------------------------------------------------------------------

def emb_id_to_name(emb_id, tokenizer):

    emb_name_utf8 = tokenizer.decoder.get(emb_id)

    if emb_name_utf8 != None:
        byte_array_utf8 = bytearray([tokenizer.byte_decoder[c] for c in emb_name_utf8])
        emb_name = byte_array_utf8.decode("utf-8", errors='backslashreplace')
    else:
        emb_name = '!Unknown ID!'

    return emb_name # return embedding name for embedding ID

#-------------------------------------------------------------------------------

def get_embedding_info(text):

    text = text.lower()

    tokenizer, internal_embs, loaded_embs = get_data()

    loaded_emb = loaded_embs.get(text, None)

    if loaded_emb == None:
        for k in loaded_embs.keys():
            if text == k.lower():
                loaded_emb = loaded_embs.get(k, None)
                break

    if loaded_emb!=None:
        emb_name = loaded_emb.name
        emb_id = '['+loaded_emb.checksum()+']' # emb_id is string for loaded embeddings
        emb_vec = loaded_emb.vec
        return emb_name, emb_id, emb_vec, loaded_emb #also return loaded_emb reference

    # support for #nnnnn format
    val = None
    if text.startswith('#'):
        try:
            val = int(text[1:])
            if (val<0) or (val>=internal_embs.shape[0]): val = None
        except:
            val = None

    # obtain internal embedding ID
    if val!=None:
        emb_id = val
    else:
        emb_ids = text_to_emb_ids(text, tokenizer)
        if len(emb_ids)==0: return None, None, None, None
        emb_id = emb_ids[0] # emb_id is int for internal embeddings

    emb_name = emb_id_to_name(emb_id, tokenizer)
    emb_vec = internal_embs[emb_id].unsqueeze(0)

    return emb_name, emb_id, emb_vec, None # return embedding name, ID, vector

#-------------------------------------------------------------------------------

def do_inspect(text):

    text = text.strip().lower()
    if (text==''): return 'Need embedding name or embedding ID as #nnnnn'

    # get the embedding info for first token in text
    emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(text)
    if (emb_name==None) or (emb_id==None) or (emb_vec==None):
        return 'An error occurred'

    results = []

    # add embedding info to results
    results.append('Embedding name: "'+emb_name+'"')
    if type(emb_id)==int:
        results.append('Embedding ID: '+str(emb_id)+' (internal)')
    else:
        results.append('Embedding ID: '+str(emb_id)+' (loaded)')

    if loaded_emb!=None:
        results.append('Step: '+str(loaded_emb.step))
        results.append('SD checkpoint: '+str(loaded_emb.sd_checkpoint))
        results.append('SD checkpoint name: '+str(loaded_emb.sd_checkpoint_name))

    vec_count = emb_vec.shape[0]
    vec_size = emb_vec.shape[1]
    results.append('Vector count: '+str(vec_count))
    results.append('Vector size: '+str(vec_size))
    results.append(SEP_STR)

    # add all vector infos to results
    tokenizer, internal_embs, loaded_embs = get_data()
    all_embs = internal_embs.to(device='cpu',dtype=torch.float32)# all internal embeddings copied to cpu as float32

    torch.set_printoptions(threshold=VEC_SHOW_TRESHOLD,profile=VEC_SHOW_PROFILE)

    """
    with open('emb_vec.txt', 'w') as f:
        f.write(str(emb_vec))
        f.close()
    """

    for v in range(vec_count):

        vec_v = emb_vec[v].to(device='cpu',dtype=torch.float32)

        # add tensor values to results

        results.append('Vector['+str(v)+'] = '+str(vec_v))
        results.append('Magnitude: '+str(torch.linalg.norm(vec_v).item()))
        results.append('Min, Max: '+str(torch.min(vec_v).item())+', '+str(torch.max(vec_v).item()))


        # calculate similar embeddings and add to results
        if vec_v.shape[0]!=internal_embs.shape[1]:
            results.append('Vector size is not compatible with current SD model')
            continue

        results.append('')
        results.append("Similar embeddings:")
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        scores = cos(all_embs, vec_v)
        sorted_scores, sorted_ids = torch.sort(scores, descending=True)
        best_ids = sorted_ids[0:MAX_SIMILAR_EMBS].numpy()
        r = []
        for i in range(0, MAX_SIMILAR_EMBS):
            emb_id = best_ids[i].item()
            emb_name = emb_id_to_name(emb_id, tokenizer)
            r.append(emb_name+'('+str(emb_id)+')')
        results.append('   '.join(r))

        results.append(SEP_STR)

    return '\n'.join(results) # return info string to results textbox

#-------------------------------------------------------------------------------

def do_save(*args):

    # do some checks
    save_name = args[-3].strip()
    enable_overwrite = args[-2]
    step_text = args[-1].strip()
    concat_mode = args[-4]
    eval_txt = args[-5].strip()
    combine_mode = args[-6]
    if save_name=='':return 'Filename is empty', None

    results = []

    save_filename = 'embeddings/'+save_name+'.bin'
    file_exists = os.path.exists(save_filename)
    if (file_exists):
        if not(enable_overwrite):
            return('File already exists, overwrite not enabled, aborting save.', None)
        else:
            results.append('File already exists, overwrite is enabled')

    step_val = None
    try:
        step_val = int(step_text)
    except:
        step_val = None
        if (step_text!=''): results.append('Step value is invalid, ignoring')

    # calculate mixed embedding in tot_vec
    vec_size = None
    tot_vec = None
    for k in range(MAX_NUM_MIX):
        name= args[k].strip().lower()

        mixval = args[k+MAX_NUM_MIX]
        if (name=='') or (mixval==0): continue

        emb_name, emb_id, emb_vec, loaded_emb = get_embedding_info(name)
        mix_vec = emb_vec.to(device='cpu',dtype=torch.float32)

        if vec_size==None:
            vec_size = mix_vec.shape[1]
        else:
            if vec_size!=mix_vec.shape[1]:
                results.append('! Vector size is not compatible, skipping '+emb_name+'('+str(emb_id)+')')
                continue

        if not(concat_mode):
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
        else:
            if tot_vec==None:
                tot_vec = mix_vec*mixval
            else:
                tot_vec = torch.cat([tot_vec,mix_vec*mixval])
            results.append('> '+emb_name+'('+str(emb_id)+')'+' x '+str(mixval))

    saved_graph = None
    # save the mixed embedding
    if (tot_vec==None):
        results.append('No embeddings were mixed, nothing to save')
    else:
        #eval feautre
        if eval_txt!='':
            vec = tot_vec.clone()
            try:
                maxn = vec.shape[0]
                maxi = vec.shape[1]
                for n in range(maxn):
                    if eval_txt.startswith('='):
                        #item-wise eval
                        for i in range(maxi):
                            v = vec[n,i]
                            ve = eval(eval_txt[1:]) #strip "="
                            vec[n,i] = ve
                    else:
                        #tensor-wise eval
                        v = vec[n]
                        ve = eval(eval_txt)
                        vec[n] = ve
                tot_vec = vec
                results.append('Applied eval: "'+eval_txt+'"')
            except Exception as e:
                results.append('Error evaluating: "'+eval_txt+'" - '+str(e))

        if (combine_mode and (tot_vec.shape[0]>1)):
            results.append('combining '+str(tot_vec.shape[0])+' vectors as 1-vector')
            tot_vec = torch.sum(tot_vec,dim=0,keepdim=True)

        new_emb = Embedding(tot_vec, save_name)
        if (step_val!=None):
            new_emb.step = step_val
            results.append('Setting step value to '+str(step_val))

        try:
            new_emb.save(save_filename)
            results.append('Saved "'+save_filename+'"')

        except:
            results.append('Error saving "'+save_filename+'" (filename might be invalid)')

        results.append('Reloading all embeddings')
        sd_hijack.model_hijack.embedding_db.dir_mtime=0
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()


        if ENABLE_GRAPH:
            # save graph
            try:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                for u in range(tot_vec.shape[0]):
                    x = torch.arange(start=0,end=tot_vec[u].shape[0],step=1)
                    plt.plot(x.numpy(), tot_vec[u].numpy())
                saved_graph = fig
            except:
                saved_graph = None

    return '\n'.join(results), saved_graph  # return info string to log textbox and saved_graph

#-------------------------------------------------------------------------------

def do_listloaded():

    tokenizer, internal_embs, loaded_embs = get_data()

    results = []
    results.append('Loaded embeddings ('+str(len(loaded_embs))+'):')
    results.append('')

    for key in loaded_embs.keys():

        try:
            emb = loaded_embs.get(key)

            r = []
            r.append(str(emb.name))
            if ENABLE_SHOW_CHECKSUM:
                r.append('    ['+str(emb.checksum())+']')
            r.append('    Vectors: '+str(emb.vec.shape[0])+' x ' +str(emb.vec.shape[1]))
            if (emb.sd_checkpoint_name!=None): r.append('    Ckpt:'+str(emb.sd_checkpoint_name))
            results.append(''.join(r))

        except:
            results.append('!error!')
            continue

    return '\n'.join(results)  # return info string to textbox

#-------------------------------------------------------------------------------

def do_minitokenize(*args):

    mini_input=args[-1].strip().lower()

    mini_sendtomix = args[-2]
    concat_mode = args[-3]
    combine_mode = args[-4]
    mix_inputs = args[0:MAX_NUM_MIX]

    tokenizer, internal_embs, loaded_embs = get_data()

    results = []

    mix_inputs_list = list(mix_inputs)

    found_ids = text_to_emb_ids(mini_input, tokenizer)
    for i in range(len(found_ids)):
        idstr = '#'+str(found_ids[i])
        results.append(idstr)
        if (mini_sendtomix==True):
            if (i<MAX_NUM_MIX): mix_inputs_list[i]=idstr

    if (mini_sendtomix==True):
        concat_mode = True
        for i in range(MAX_NUM_MIX):
            if (i>=len(found_ids)): mix_inputs_list[i]=''

    combine_mode = False

    return *mix_inputs_list,concat_mode,combine_mode,' '.join(results)# return everything

#-------------------------------------------------------------------------------

def do_reset(*args):

    mix_inputs_list = [''] * MAX_NUM_MIX
    mix_slider_list = [1.0] * MAX_NUM_MIX

    return *mix_inputs_list, *mix_slider_list

#-------------------------------------------------------------------------------

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tabs():
            with gr.Row():

                with gr.Column(variant='panel'):
                    text_input = gr.Textbox(label="Text input", lines=1, placeholder="Enter embedding name (only first token is processed), or embedding ID as #nnnnn")
                    with gr.Row():
                        inspect_button = gr.Button(value="Inspect", variant="primary")
                        listloaded_button = gr.Button(value="List loaded embeddings")
                    inspect_result = gr.Textbox(label="Results", lines=15)

                    with gr.Column(variant='panel'):
                        mini_input = gr.Textbox(label="Mini tokenizer", lines=1, placeholder="Enter a short prompt (loaded embeddings or modifiers are not supported)")
                        with gr.Row():
                            mini_tokenize = gr.Button(value="Tokenize", variant="primary")
                            mini_sendtomix = gr.Checkbox(value=False, label="Send IDs to mixer")
                        mini_result = gr.Textbox(label="Tokens", lines=1)

                with gr.Column(variant='panel'):
                    with gr.Row():
                        gr.Column(variant='panel')
                        reset_button = gr.Button(value="Reset mixer")

                    mix_inputs = []
                    mix_sliders = []
                    for n in range(MAX_NUM_MIX):
                        with gr.Row():
                           with gr.Column():
                               mix_inputs.append(gr.Textbox(label="Name "+str(n), lines=1, placeholder="Enter name of embedding to mix"))
                           with gr.Column():
                               mix_sliders.append(gr.Slider(label="Multiplier",value=1.0,minimum=-1.0, maximum=1.0, step=0.1))

                    with gr.Row():
                            with gr.Column():
                                concat_mode = gr.Checkbox(value=False,label="Concat mode")
                                combine_mode =  gr.Checkbox(value=False,label="combine as 1-vector")
                            eval_box =  gr.Textbox(label="Eval",lines=2,placeholder='=v+torch.rand(1)*0.1')
                            step_box = gr.Textbox(label="Step",lines=1,placeholder='only for training')

                    with gr.Row():
                        save_name = gr.Textbox(label="Filename",lines=1,placeholder='Enter file name to save')
                        save_button = gr.Button(value="Save mixed", variant="primary")
                        enable_overwrite = gr.Checkbox(value=False,label="Enable overwrite")

                    with gr.Row():
                        save_result = gr.Textbox(label="Log", lines=10)
                        save_graph = gr.Plot()

            listloaded_button.click(fn=do_listloaded, outputs=inspect_result)
            inspect_button.click(fn=do_inspect,inputs=[text_input],outputs=[inspect_result])
            save_button.click(fn=do_save, inputs=mix_inputs+mix_sliders+[combine_mode, eval_box, concat_mode,save_name,enable_overwrite,step_box],outputs=[save_result, save_graph])

            mini_tokenize.click(fn=do_minitokenize,inputs=mix_inputs+[combine_mode, concat_mode, mini_sendtomix, mini_input], outputs=mix_inputs+[concat_mode,combine_mode, mini_result])

            reset_button.click(fn=do_reset,outputs=mix_inputs+mix_sliders)

    return [(ui, "Embedding Inspector", "inspector")]

script_callbacks.on_ui_tabs(add_tab)
