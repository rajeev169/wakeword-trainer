# Shivangi wake-word trainer

Trains **shivangi.onnx**, the model that lets the Shivangi app wake up when someone says "Shivangi".
It runs on GitHub Actions; nothing needs to be installed on your phone or computer.

## How it works
1. Generates thousands of voices saying "Shivangi" (about 900 English speakers + Hindi voices saying "शिवांगी").
2. Generates similar names (Shivani, Shivam, Sangeeta…) and everyday sentences, so she learns what is **not** her name.
3. Adds background noise (fan, hum, other people talking) to make it work in real rooms.
4. Turns every clip into speech features with the open-source openWakeWord models (in `base_models/`, Apache-2.0),
   exactly the way the app does on the phone, and trains a small network on them.
5. Exports `shivangi.onnx` (about 1 MB) plus `report.json` with the test scores.

## Run it
1. Create a new GitHub repository (for example `shivangi-wakeword`) and upload everything from this zip, including the hidden `.github` folder.
2. Open the repo → **Actions** → **Train Shivangi wake word** → **Run workflow** (keep the defaults).
3. Wait about 1–2 hours. When it is green, open the run and download the **shivangi-wakeword** artifact.
4. Inside it is `shivangi.onnx`. Put it in the app project at `app/src/main/assets/wake/shivangi.onnx` and build the app again.

## Reading report.json
- `val_recall`: how often she wakes when "Shivangi" is said (higher is better, aim for 0.9+).
- `val_false_positive_rate`: how often she wakes for other speech (lower is better, aim for under 0.01).
If false wakes are too high, run again with more `samples`; the Sensitivity slider in the app also helps.

## Changing the name
Edit the texts in the workflow steps "Generate Shivangi" and "Generate शिवांगी", and the similar words in
`adversarial_en.txt` / `adversarial_hi.txt`.

Credits: openWakeWord (Apache-2.0), piper-sample-generator (MIT), Piper voices (see their model cards).
