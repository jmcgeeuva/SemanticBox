import math

def run_example(generated_images, processor, model, real_texts, bbs=None, label=None, text_input=None, debug=False, test=False):

    REGION_TO_DESCRIPTION = 'REGION_TO_DESCRIPTION'
    CAPTION = 'CAPTION'

    prompt_type = f'<{CAPTION}>' #'<DETAILED_CAPTION>'
    prompt_cap = [prompt_type for _ in generated_images]

    # Just choose one frame to create a caption for
    if not test2:
        cropped_images = [transforms.ToPILImage()(images[0]) for images in generated_images]
    else:
        cropped_images = [transforms.ToPILImage()(images[random.randint(0, len(images)-1)]) for images in generated_images]
    
    bounded_text = []
    if bbs != None:
        # [16, 8, 4] = [B, T, C]
        prompts_r2d = []
        bbs = bbs[:, 0, :]
        width = cropped_images[0].size[0]
        height = cropped_images[0].size[1]
        for bb in bbs:
            norm_bb = normalize(bb.tolist(), width, height)
        
            prompt_r2d = f'<{REGION_TO_DESCRIPTION}>'
            for dim in norm_bb:
                prompt_r2d += f'<loc_{dim}>'
            prompts_r2d.append(prompt_r2d)
        
        generated_texts = []
        text_dict = {
            CAPTION: [],
            REGION_TO_DESCRIPTION: []
        }
        for task, prompt, padding in [(CAPTION, prompt_cap, False), (REGION_TO_DESCRIPTION, prompts_r2d, True)]:
            inputs = processor(text=prompt, images=cropped_images, return_tensors="pt", padding=padding)

            input_ids = inputs["input_ids"].to(model.module.device)
            pixel_values = inputs["pixel_values"].to(model.module.device)
            generated_ids = model.module.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=512,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for this_text, this_prompt in zip(generated_text, prompt):
                parsed_answer = processor.post_process_generation(
                    this_text,
                    task=this_prompt,
                    image_size=(width, height)
                )
                text_dict[task].append(parsed_answer[this_prompt])

        
        generated_text = text_dict[CAPTION]
        bounded_text = text_dict[REGION_TO_DESCRIPTION]
        return generated_text, bounded_text
    else:
        inputs = processor(text=prompt_cap, images=cropped_images, return_tensors="pt", padding=False)

        input_ids = inputs["input_ids"].to(model.module.device)
        pixel_values = inputs["pixel_values"].to(model.module.device)
        generated_ids = model.module.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=512,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
        return generated_text, None

def unnormalize(bb, w, h):
    """unnormalize
    Changes the florence style bounding box into the picture's scale
    """
    new_bb = []
    for i, b in enumerate(bb):
        if i % 2 != 0:
            new_bb.append((b/1000)*h)
        else:
            new_bb.append((b/1000)*w)
    return new_bb

def normalize(bb, w, h):
    """normalize
    Changes the bounding box given into florence style bounding boxes
    """
    new_bb = []
    for b in bb:
        if b % 2 != 0:
            new_bb.append(math.ceil((b/h)*1000))
        else:
            new_bb.append(math.ceil((b/w)*1000))
    return new_bb