import os 
import io
from sys import api_version
# from .Google import Create_Service
# from googleapiclient.http import MediaIoBaseDownload

from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth.transport.requests import Request


def Create_Service(client_secret_file, api_name, api_version, *scopes):
    print(client_secret_file, api_name, api_version, scopes, sep='-')
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]
    print(SCOPES)

    cred = None

    pickle_file = f'token_{API_SERVICE_NAME}_{API_VERSION}.pickle'
    # print(pickle_file)

    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            cred = flow.run_local_server()

        with open(pickle_file, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=cred)
        print(API_SERVICE_NAME, 'service created successfully')
        return service
    except Exception as e:
        print('Unable to connect.')
        print(e)
    return None

# def Create_Service(client_secret_file, api_name, api_version, *scopes):
#     print(client_secret_file, api_name, api_version, scopes, sep='-')
#     CLIENT_SECRET_FILE = client_secret_file
#     API_SERVICE_NAME = api_name
#     API_VERSION = api_version
#     SCOPES = [scope for scope in scopes[0]]
#     print(SCOPES)

CLIENT_SECRET_FILE = 'Download_AeroBOT_data.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

# file_ids = ['1HZSxIfwGqg38yByiXxS4EjyLgeHI6yOv']
# file_names = ['df_test_for_Anomaly_prediction.pkl']

# for file_id, dile_name in zip(file_ids, file_names):
#     request = service.Files().get_media(fileId=file_id)

#     fh = io.BytesIO()
#     downloader = MediaIoBaseDownload(fd=fh, request=request)

#     done = False

#     while not done: 
#         status, done = downloader.next_chunk()
#         print('Download progress {0'.format(status.progress() * 100))

#         fh.seek(0)

#         with open(os.path.join('./data/downloaded_from_GDrive'), 'wb') as f:
#             f.write(fh.read())
#             f.close()