# Caesb Control

## TL;DR
Sistema para monitorar contas de água da Caesb. Detecta anomalias de consumo usando IA.

**Início Rápido:**
```bash
git clone https://github.com/roneiFonseca/CaesbControl.git
cd CaesbControl
python -m venv venv
.\venv\Scripts\activate
pip install -e .
python manage.py migrate
python manage.py runserver
```
Acesse http://localhost:8000 (usuário: admin, senha: admin123)

## Funcionalidades

- Gerenciamento de unidades de consumo (imóveis)
- Gerenciamento de contas de água
- Análise de anomalias em contas
- API RESTful com documentação Swagger
- Autenticação via JWT (JSON Web Token)
- Dashboard com gráficos e estatísticas
- Upload de imagens/PDFs das contas
- Suporte para múltiplas unidades de consumo

## Requisitos

- Python 3.8+
- uv (gerenciador de pacotes Python)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/roneiFonseca/CaesbControl.git
cd CaesbControl
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -e .
```
Edite o arquivo `.env` e adicione suas chaves de API:
- `OPENAI_API_KEY`: Chave da API do OpenAI (para análise de anomalias)

4. Execute as migrações:
```bash
python manage.py migrate
```

5. Execute o servidor de desenvolvimento:
```bash
python manage.py runserver
```

O sistema estará disponível em http://localhost:8000

## Uso

### Interface Web

Acesse http://localhost:8000 e faça login com suas credenciais.

### API REST

A documentação da API está disponível em:
- Swagger UI: http://localhost:8000/api/docs/
- ReDoc: http://localhost:8000/api/redoc/

A API usa autenticação JWT (JSON Web Token). Para usar a API:

1. **Obter Token**:
***Já existe um usuário cadastrado: admin/admin123***
```bash
# Substitua usuario e senha pelos seus dados
curl -X POST http://localhost:8000/api/token/ \
  -H "Content-Type: application/json" \
  -d '{"username":"admin", "password":"admin123"}'

# Resposta:
{
    "access": "seu_token_jwt",
    "refresh": "seu_token_refresh"
}
```

2. **Usar o Token**:
```bash
curl -X GET http://localhost:8000/api/properties/ \
  -H "Authorization: Bearer seu_token_jwt"
```

3. **Renovar Token**:
```bash
curl -X POST http://localhost:8000/api/token/refresh/ \
  -H "Content-Type: application/json" \
  -d '{"refresh":"seu_token_refresh"}'
```

4. **Verificar Token**:
```bash
curl -X POST http://localhost:8000/api/token/verify/ \
  -H "Content-Type: application/json" \
  -d '{"token":"seu_token_jwt"}'
```

5. **No Swagger UI**:
   - Clique em "Authorize"
   - Digite: `Bearer seu_token_jwt`
   - Clique em "Authorize"

### Endpoints Principais

- `/api/properties/`: Gerenciamento de unidades de consumo
- `/api/bills/`: Gerenciamento de contas de água
- `/api/explain-bill/{property_id}/{year}/`: Análise de anomalias em contas

## Interface Web

- Dashboard: http://localhost:8000/
- Admin: http://localhost:8000/admin/

## Desenvolvimento

- Django 5.1
- Django REST Framework
- SQLite como banco de dados
- Bootstrap 5 para o frontend
- drf-yasg para documentação Swagger
- APIs externas:
  - OpenAI: Análise de anomalias em contas
  - Serper: Busca web para contexto
  - Groq: Processamento de linguagem natural

## Licença

Apenas Para Testes.
