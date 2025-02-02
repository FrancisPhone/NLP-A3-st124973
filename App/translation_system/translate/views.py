from django.shortcuts import render
from .utils import translate_sentence, gen_model, mul_model, add_model, source_tokenizer, target_tokenizer
from .syllable_break import syllable_break
from .models import TranslationHistory
from datetime import datetime

segmentation = syllable_break()


def translate_text(request):
    # Get the source sentence from the request
    source_text = request.GET.get("source_text", "")
    translations = list()
    models = ['General Attention', 'Multiplicative Attention', 'Additive Attention']

    if source_text:
        source_text = ' '.join(segmentation.syllable_break(source_text))
        for i, model in enumerate([gen_model, mul_model, add_model]):
            # Call your translation function (as defined in the previous code)
            translation, attentions = translate_sentence(source_text, model, source_tokenizer, target_tokenizer)
            translations.append(translation)
            save_translation_history(source_text, translation, models[i])
    # Return the translated text and attentions as JSON response
    history = TranslationHistory.objects.all().order_by('-timestamp')
    return render(request, 'index.html',
                  {'generated_results': zip(models, translations), 'history': history})


def save_translation_history(source_text, translated_text, model_name):
    # Save translation details to the database
    TranslationHistory.objects.create(
        source_text=source_text,
        translated_text=translated_text,
        model_name=model_name,
        timestamp=datetime.now()  # You can also let Django handle this automatically
    )
