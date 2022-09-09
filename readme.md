## DeepFish
World of Warcraft fishing bot powered by Computer Vision and Deep Learning.

![DeepFish logo](images/logo.png)

Note: Usage of this bot is abuse of World of Warcraft terms of service. Please, don't use it to abuse fishing.

## Showcase 
All actions on this video are being performed by deepfish bot.

https://user-images.githubusercontent.com/60406311/187909093-f8aec3ec-73fa-4341-9ce8-1bacab40f27d.mp4


### Key features
1. Platforms: Windows (tested on Windows 10)
2. Runs on CPU.
3. No memory modification (it's sneaky to anti-cheats!)
4. Well-tested and should work both on Classic and Retail version.


### Installation
```commandline
git clone https://github.com/Datasciensyash/deepfish.git
cd deepfish
pip install .
```

Note: Use new environment to install this bot.

### Usage

Start bot with following command:

```commandline
run_deepfish_bot -k FISHING_KEY
```

And if you don't want to see logs, add `--suppress_logging` flag:

```commandline
run_deepfish_bot -k FISHING_KEY --supress_logging
```

Where `FISHING_KEY` is key assigned to a fishing skill, e.g. `0`.

### Authors

Bot & Model training - Datasciensyash

Dataset gathering & Model training - LisenKapusta
