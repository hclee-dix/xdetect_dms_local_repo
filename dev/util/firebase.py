from dev.core.firebase import firebase_app

# def uploadFile(source_file,project_id,model_id,destination_file_name='output',file_type="image/jpeg"):
def uploadFile(src,dst):
    """
        download_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{urllib.parse.quote(path)}?alt=media"
    """
    blob = firebase_app.bucket.blob(dst)
    blob.upload_from_filename(src,content_type=blob.content_type)
    blob.make_public()
    download_url = blob.public_url
    return download_url