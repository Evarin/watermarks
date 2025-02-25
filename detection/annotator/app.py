from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from pathlib import Path
from sqlalchemy import create_engine, text

DATA_DIR = Path("../../data")

app = Flask(__name__)

engine = create_engine('sqlite:///db.sqlite3')

# Create a table ANNOTATIONS with columns id, image, x0, y0, x1, y1
with engine.connect() as connection:
    connection.execute(text('''
    CREATE TABLE IF NOT EXISTS ANNOTATIONS (
        image TEXT PRIMARY KEY,
        x0 REAL, y0 REAL,
        x1 REAL, y1 REAL
    )
    '''))


@app.route('/')
@app.route('/<path:path>', methods=["GET", "POST"])
def annotator(path=""):
    path = Path(path)
    all_files = os.listdir(DATA_DIR / path)

    # list all subdirectories
    subdirs = sorted([path / f for f in all_files if os.path.isdir(DATA_DIR / path / f)])
    if path != Path(""):
        subdirs = [path / ".."] + subdirs

    # list all images
    images = [path / f for f in all_files if os.path.isfile(DATA_DIR / path / f)]
    images = sorted([f for f in images if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])

    cur_image = 0
    image = None
    prev_image = None
    next_image = None
    annotations = None

    if images:
        # find the current image
        if request.args.get("image"):
            cur_image = int(request.args.get("image"))
        prev_image = (cur_image - 1) % len(images)
        next_image = (cur_image + 1) % len(images)

        image = images[cur_image]

        if request.method == "POST":
            # save annotation to database
            x0 = float(request.form.get("bbx0"))
            y0 = float(request.form.get("bby0"))
            x1 = float(request.form.get("bbx1"))
            y1 = float(request.form.get("bby1"))

            # upsert with sqlalchemy 
            with engine.connect() as connection:
                connection.execute(text('''
                    INSERT INTO ANNOTATIONS (image, x0, y0, x1, y1) VALUES (:image, :x0, :y0, :x1, :y1)
                    ON CONFLICT(image) DO UPDATE SET x0=:x0, y0=:y0, x1=:x1, y1=:y1
                '''), {"image":str(image), "x0":x0, "y0":y0, "x1":x1, "y1":y1})
                connection.execute(text('COMMIT'))
            
            # redirect to next image
            return redirect(url_for('annotator', path=path) + "?image=" + str(next_image))

        # get: fetch annotation from database
        with engine.connect() as connection:
            annotations = connection.execute(text('SELECT * FROM ANNOTATIONS WHERE image = :image'), {"image":str(image)}).fetchone()

    with engine.connect() as connection:
        # count where startswith path
        count_annotations = connection.execute(text('SELECT COUNT(*) FROM ANNOTATIONS WHERE image LIKE :path'), {"path":str(path / "%")}).fetchone()[0]

    return render_template('index.html', 
                           subdirs=subdirs, image=image, cur_path=path,
                           len_images=len(images), cur_image=cur_image, count_annotations=count_annotations,
                           prev_image=prev_image, next_image=next_image, annotations=annotations)

@app.route('/media/<path:filename>')
def media(filename):
    # serve the file from the data directory
    return send_from_directory(
        DATA_DIR,
        filename,
        as_attachment=True
    )

if __name__ == '__main__':
    app.run(debug=True)