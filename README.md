# dataquitaine2023
Code pour la présentation de Dataquitaine 2023

## FLUE CLS

Code pour l'analyse de sentiments :

- cls_finetuning.py : Finetuning d'un modèle pour de la classification binaire

- cls_prompt_mt0.py : Analyse de sentiments utilisant un prompt fixe avec un modèle mT0.

- cls_prompt_simple.py : Analyse de sentiments en utilisant un prompt simple et la librarie OpenPrompt adaptée pour les modèles français (<https://github.com/catie-aq/OpenPrompt/tree/french_prompts>).

- cls_softprompt.py : Analyse de sentiments avec un prompt automatique (PTuning) et la librarie OpenPrompt.

## NER

- ner_finetuning.py : Finetuning d'un modèle sur un problème de NER

- ner_prompt_mt0.py : NER utilisant des prompts et un modèle mT0

- ner_prompt.py : NER avec des prompts simples "mots à mots"

- qa_ner_prompt.py : NER avec un modèle de Question-answering (<https://huggingface.co/CATIE-AQ/QAmembert>).

## OrangeSum

- orange-sum-finetuning.py : Génération de titre par finetuning

- orange-sum-mt0.py: Génération de titres avec un prompt mT0.