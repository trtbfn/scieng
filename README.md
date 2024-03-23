# Science Papers Semantic Search

1. You should run docker container with Groubid. Instruction: https://grobid.readthedocs.io/en/latest/Grobid-docker/

```bash
docker pull lfoppiano/grobid:0.8.0
docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0
```

2. Download groubit client: https://github.com/kermitt2/grobid_client_python

3. Run libraries installation:

```bash
pip install -r requirements.txt
``` 
