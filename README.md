# Chatbot conectado a YouTube (LangChain + Streamlit)

Agente de IA conectado a YouTube que permite chatear con el contenido de los videos.

![image](https://github.com/sergiovogt/youtube_chatbot_streamlit/assets/159809335/0d22419c-ea1f-458e-bc5a-1f1c9f6ccad5)

## Características

- Conecta videos de YouTube con una IA para chatear con el contenido.
- Desarrollado con LangChain y Streamlit.

## Instalación

#### 1. Clonar el repositorio

```bash
git clone https://github.com/sergiovogt/youtube_chatbot_streamlit.git
```

#### 2. Crear el entorno

``` bash
cd youtube_chatbot_streamlit
python -m venv env
source env/bin/activate
```

#### 3. Instalar las dependencias requeridas
``` bash
pip install -r requirements.txt
```

Primero, crear el archvio `.env` en el directorio raiz del proyecto. Dentro del archvio, agreagar la API Key de OpenAI:

```makefile
OPENAI_API_KEY="agregar_aquí_la_apikey"
```

Guardar el archivo y cerrarlo. En el script de Python, cargar el archivo `.env` usando el siguiente código (ya está cargado en [app.py]:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

Ahora tu entorno de Python está listo! ya puedes continuar...

## Tutoriales
Para ver más tutoriales, podés visitar mi canal de YouTube:  [youtube.com/@sergiovogtds1998](https://youtube.com/@sergiovogtds1998)
