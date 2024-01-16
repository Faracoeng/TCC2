#Trazer das variaveis de ambiente

class FastAPI:
    
    host = "http://127.0.0.1:5000/"

    headers = {
        'Authorization': 'Token b389fa4f1ac145cad4e1a4ddcde2a879b6881c96',
        'Content-Type': 'application/json'
    }

    @classmethod
    def set_host(cls, host, headers):
        cls.host = host
        cls.headers = headers