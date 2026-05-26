# DSRE EX / Reborned DSRE (Deep Sound Resolution Enhancer)

<img width="640" height="380" alt="img" src="https://github.com/user-attachments/assets/eda2aabf-e756-4da5-9d0b-8e52f6133b12" />

DSRE EX is a high-performance audio enhancement tool that can batch-convert any audio files into high-resolution (Hi-Res) audio. Inspired by Sony DSEE HX, it uses a non-deep-learning frequency enhancement algorithm, allowing fast processing of large batches without heavy computation.

## Histroy

● DSRE

It was originally developed by Chinese developer Qu Le fan(屈乐凡) and, as is well known, began as an audio upscaler project inspired by DSEE.
Unfortunately, the developer is no longer maintaining it, and it has become a dead project.


● Ver 2.0 (EN)

https://github.com/Urabewe/DSRE---Digital-Sound-Resolution-Enhancer-English
It was known as the only forked project with modifications to the English version, but since this version has had the core upscaling logic removed by AI, it is not technically a DSRE.
This is because the upscaling code has been completely replaced by EQ-based sound effects and plugin chains.

● Ver 3.x (KR)

https://arca.live/b/breaking/165547998
A user known as Noir16(느와르), who is now believed to be inactive, referenced the original DSRE and EN version GUI code to create a version that was re-maintained to be closer to DSEE. the original project’s goal. And was first released on arca.live, a well-known Korean community.

Although this user did not use GitHub, the source code was distributed alongside the release on that community.

Differences from the origin: 

-- The GUI from Ver 2.0 (EN) has been implemented.

-- Using ARDFTSRC as the resampler.

-- There are no parameters. settings are adjusted automatically.

-- It incorporates psychoacoustic technology that serves a similar purpose to DSEE HX but uses a different approach.

-- Multilingual support for 4 languages.

Upon reviewing the source code, it was confirmed that, like the EN version, AI was used, and the code was written to ensure that the existing code remained intact, with the exception of the GUI.


This fork was created to maintain the discontinued Ver 3.x. You can view the changes on the release page.

## How do I build it?

1. Install Python 3.10.11.

2. Creating a Virtual Environment `python -m venv dsre_env`, and settings `dsre_env\Scripts\activate`

3. Install the required package `pip install -r requirements.txt`, and upgrade pip `python -m pip install --upgrade pip`

4. Test `python DSRE.py`

5. build `pyinstaller --onefile --windowed --add-data "logo.ico;." --add-binary "ffmpeg.exe;." --icon=logo.ico --name=DSRE DSRE.py`

※ To run and build this, you'll need ffmpeg.exe, which must be located in the same directory as the .py file.

## TODO

- [x] Fix the automatic parameter calculation formula so that it does not reference the resampling space.
- [ ] Fix the issue where sound quality was degraded in certain frequency bands.
- [ ] Imports a MacOS GUI from [CrossDarkrix/DSRE-Audio-Enhancer](https://github.com/CrossDarkrix/DSRE-Audio-Enhancer).

## Credits
- DSRE: https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer
- DSRE-English (Forked-GUI): https://github.com/Urabewe/DSRE---Digital-Sound-Resolution-Enhancer-English
- DSRE-v3.5 (Forked-Rebase): https://arca.live/b/breaking/165547998
- ARDFTSRC: https://github.com/mrspoonsi/ARDFTSRC
- FFmpeg: https://ffmpeg.org/

SOME PART OF THIS SOURCE CODE WERE GENERATED USING AI (X.com Grok & Anthropic Claude).
THIS IMPLIES THAT AI CODE GENERATORS WAS USED, BUT IT DOES NOT MEAN THAT
THE PROGRAM ITSELF OPERATES USING AI
