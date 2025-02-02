from django.db import models


class TranslationHistory(models.Model):
    source_text = models.TextField()
    translated_text = models.TextField()
    model_name = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Translation from {self.source_text[:30]}... using {self.model_name} at {self.timestamp}"

