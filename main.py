import os
import json
import pandas as pd
from copy import deepcopy
from zipfile import ZipFile
from pydantic import BaseModel, Field

# OpenAI client initialization
from openai import OpenAI
client_openai = OpenAI()

MODEL_NAME = "gpt-4o-mini"

# Hyperparameters
TRACK = "track_b"  # Options: track_a, track_b, track_c
TARGET_LANGUAGES = ["chn"]  # Specify languages for processing
INPUT_DIR = os.path.join("public_data_test", TRACK, "test")
OUTPUT_DIR = f"{TRACK}_output_chn"
FINAL_ZIP = f"submission_{TRACK}_chn.zip"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion categories (including disgust for all, with special handling for English)
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

class EmotionRewrittenText(BaseModel):
    original_text: str
    joy: str
    sadness: str
    fear: str
    anger: str
    surprise: str
    disgust: str

class EmotionAnalysisNonNestedOutput(BaseModel):
    joy_reasoning: str
    joy_intensity: int
    sadness_reasoning: str
    sadness_intensity: int
    fear_reasoning: str
    fear_intensity: int
    anger_reasoning: str
    anger_intensity: int
    surprise_reasoning: str
    surprise_intensity: int
    disgust_reasoning: str
    disgust_intensity: int

# System prompts
PROMPT_REWRITE = """You are a helpful language model tasked with rewriting a given text to convey specific emotional tones. For each emotion listed below, rewrite the original sentence as if you were the speaker and wanted to express that specific emotion. Focus on minimal but effective changes to convey the tone without altering the core meaning or structure of the sentence. Ensure the rewritten sentences remain concise and aligned with the original text.

Emotions to rewrite for:
- Joy
- Sadness
- Fear
- Anger
- Surprise
- Disgust

Please rewrite the text for each emotion. Do not extend or shorten the text unnecessarily."""

PROMPT_SCORE = """You are a helpful language model tasked with analyzing the emotional intensity of an original sentence. Given a set of rewritten sentences (one for each emotion), evaluate the original sentence to determine how strongly it aligns with each emotion. The rewritten sentences are clues to help guide your assessment, but your focus should remain on the original sentence.

Instructions:
1. For each emotion (joy, sadness, fear, anger, surprise, disgust), compare the original sentence to its corresponding rewritten version.
2. Assess how closely the original sentence aligns with the tone of each rewritten sentence, considering subtle cues in language, context, and implied sentiment.
3. Provide a brief reasoning for each emotion explaining the alignment.
4. Assign an intensity score for each emotion:
   - 0: No emotion present
   - 1: Low degree of emotion
   - 2: Moderate degree of emotion
   - 3: High degree of emotion

Provide your analysis and intensity scores for each emotion."""

def rewrite_into_emotions(question):
    messages = [
        {"role": "system", "content": PROMPT_REWRITE},
        {"role": "user", "content": question}
    ]
    try:
        extract = client_openai.beta.chat.completions.parse(
            messages=messages,
            model=MODEL_NAME,
            response_format=EmotionRewrittenText
        )
        output = json.loads(extract.choices[0].message.content)
    except Exception as e:
        print(f"Error in rewrite_into_emotions: {e}")
        output = {
            "original_text": question,
            "joy": "",
            "sadness": "",
            "fear": "",
            "anger": "",
            "surprise": "",
            "disgust": ""
        }
    return {"messages": messages, "output": output}

def emotion_score(original_text, rewritten_forms, is_english=False):
    """
    Evaluates the emotional intensity of an original sentence based on rewritten forms.
    """
    messages = [
        {"role": "system", "content": PROMPT_SCORE},
        {"role": "user", "content": json.dumps({
            "original_text": original_text,
            "rewritten_form_joy": rewritten_forms.get("joy", ""),
            "rewritten_form_sadness": rewritten_forms.get("sadness", ""),
            "rewritten_form_fear": rewritten_forms.get("fear", ""),
            "rewritten_form_anger": rewritten_forms.get("anger", ""),
            "rewritten_form_surprise": rewritten_forms.get("surprise", ""),
            "rewritten_form_disgust": rewritten_forms.get("disgust", "")
        })}
    ]
    
    try:
        extract = client_openai.beta.chat.completions.parse(
            messages=messages,
            model=MODEL_NAME,
            response_format=EmotionAnalysisNonNestedOutput
        )
        output = json.loads(extract.choices[0].message.content)

        # Handle disgust for English (set to None if missing)
        if is_english:
            output["disgust_reasoning"] = ""
            output["disgust_intensity"] = 0
    except Exception as e:
        print(f"Error in emotion_score: {e}")
        output = {
            "joy_reasoning": "",
            "joy_intensity": 0,
            "sadness_reasoning": "",
            "sadness_intensity": 0,
            "fear_reasoning": "",
            "fear_intensity": 0,
            "anger_reasoning": "",
            "anger_intensity": 0,
            "surprise_reasoning": "",
            "surprise_intensity": 0,
            "disgust_reasoning": "",
            "disgust_intensity": 0
        }
    
    return {"messages": messages, "output": output}

def predict_emotions(row, language_code):
    ans_dict = deepcopy(row).to_dict()
    question = row.get("text", "")

    # Step 1: Rewrite into emotions
    step_1 = rewrite_into_emotions(question)
    rewritten_forms = step_1.get("output", {})

    ans_dict.update({
        "rewritten_joy": rewritten_forms.get("joy", ""),
        "rewritten_sadness": rewritten_forms.get("sadness", ""),
        "rewritten_fear": rewritten_forms.get("fear", ""),
        "rewritten_anger": rewritten_forms.get("anger", ""),
        "rewritten_surprise": rewritten_forms.get("surprise", ""),
        "rewritten_disgust": rewritten_forms.get("disgust", "")
    })

    # Step 2: Score emotions
    step_2 = emotion_score(question, rewritten_forms, language_code == "eng")
    emotion_scores = step_2.get("output", {})

    for emotion in EMOTION_LABELS:
        ans_dict[f"reasoning_{emotion}"] = emotion_scores.get(f"{emotion}_reasoning", "")
        ans_dict[f"pred_{emotion}"] = emotion_scores.get(f"{emotion}_intensity", 0)

    # Save intermediate results for debugging
    intermediate_dir = os.path.join(OUTPUT_DIR, f"intermediate_{language_code}")
    os.makedirs(intermediate_dir, exist_ok=True)

    try:
        with open(os.path.join(intermediate_dir, f"step_1_{row['id']}.json"), "w") as f:
            json.dump({"messages": step_1.get("messages", []), "output": rewritten_forms}, f, indent=4)
    except Exception as e:
        print(f"Error saving step 1 results for {row['id']}: {e}")

    try:
        with open(os.path.join(intermediate_dir, f"step_2_{row['id']}.json"), "w") as f:
            json.dump({"messages": step_2.get("messages", []), "output": emotion_scores}, f, indent=4)
    except Exception as e:
        print(f"Error saving step 2 results for {row['id']}: {e}")

    return ans_dict


def process_language(language_code):
    input_file = os.path.join(INPUT_DIR, f"{language_code}.csv")
    if not os.path.exists(input_file):
        print(f"Input file for language {language_code} not found. Skipping.")
        return

    data = pd.read_csv(input_file)
    results = []

    for _, row in data.iterrows():
        try:
            print(f"Processing {row['id']} in {language_code}")
            results.append(predict_emotions(row, language_code))
        except Exception as e:
            print(f"Error processing row {row['id']} in {language_code}: {e}")

    result_df = pd.DataFrame(results)

    # Convert intensities >1 to 1 for tracks A and C
    if TRACK in ["track_a", "track_c"]:
        for emotion in EMOTION_LABELS:
            result_df[f"pred_{emotion}"] = result_df[f"pred_{emotion}"].apply(lambda x: 1 if x > 0 else 0)

    # Handle English-specific removal of disgust column
    if language_code == "eng":
        result_df = result_df.drop(columns=["rewritten_disgust", "reasoning_disgust", "pred_disgust"], errors="ignore")

    # Save processed results
    try:
        result_file = os.path.join(OUTPUT_DIR, f"pred_{language_code}.csv")
        result_df.to_csv(result_file, index=False)
    except Exception as e:
        print(f"Error saving results for {language_code}: {e}")

# Main processing loop
for lang in TARGET_LANGUAGES:
    try:
        process_language(lang)
    except Exception as e:
        print(f"Error processing language {lang}: {e}")

# Create a zip file for submission
try:
    with ZipFile(FINAL_ZIP, 'w') as zipf:
        for file in os.listdir(OUTPUT_DIR):
            zipf.write(os.path.join(OUTPUT_DIR, file), arcname=file)
    print(f"Submission file created: {FINAL_ZIP}")
except Exception as e:
    print(f"Error creating submission file: {e}")