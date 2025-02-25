# A very basic bounding box annotator

This Flask app displays every image in the directory `DATA_DIR` (by default, `../../data/`), and allows the user to draw the bounding box of *one* watermark per file.

The bounding boxes are saved in a `db.sqlite` file.

## How to use

1. Put the images in the `DATA_DIR` directory. The images should be in the `jpg` or `png` format. They can be in subdirectories.
2. Run the app with `python app.py`.
3. Open your web browser and go to `http://localhost:5000`.
4. Draw the bounding box of the watermark in the image by clicking on the top-left corner, then the bottom-right corner, and save (you can press enter to do so).