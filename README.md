Projet LLM - Ollama

Description
Ce projet permet de télécharger un fichier depuis Google Drive et de l'utiliser avec Ollama pour discuter avec l'IA du contenu du fichier. Cela peut inclure des fichiers PDF, des documents texte et des fichiers JSON.

Prérequis
Assurez-vous d'avoir Python installé sur votre machine. Ce projet a été testé avec Python 3.12.

Setup :

- pip install -r requirements.txt
- Installer Ollama (https://ollama.com/download)
- ollama pull llama3
- ollama pull mxbai-embed-large

Utilisation :

- python upload.py
- python localrag.py

Configuration de la créativité d'Ollama :

- ollama run llama3
- /set parameter temperature <valeur>
