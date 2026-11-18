import subprocess, sys
SEG = {
 "en": [(0.0,1.43,0.9),(1.81,3.86,2.45),(4.31,5.04,4.5),(5.31,6.11,6.1),(6.36,7.12,7.7),
        (7.60,14.35,12.5),(14.77,16.58,20.4),(16.96,20.55,22.4),(21.01,23.57,27.6),
        (23.99,25.83,33.6),(26.22,28.76,35.8)],
 "ar": [(0.0,1.51,0.9),(1.96,3.39,2.5),(3.80,4.49,4.5),(4.75,5.42,6.1),(5.69,6.53,7.7),
        (6.90,14.71,12.2),(15.07,17.44,20.4),(17.73,22.20,22.85),(22.54,25.23,27.6),
        (25.59,28.58,33.4),(28.86,31.95,36.6)],
}
lang, out = sys.argv[1], sys.argv[2]
segs = SEG[lang]; f = []; labels = []
for i,(a,b,at) in enumerate(segs):
    a2, b2 = max(0,a-0.06), b+0.12
    d = int(at*1000)
    f.append(f"[1:a]atrim={a2}:{b2},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,afade=t=out:st={b2-a2-0.08}:d=0.08,aresample=48000,aformat=channel_layouts=stereo,adelay={d}|{d}[s{i}]")
    labels.append(f"[s{i}]")
f.append("".join(labels)+f"amix=inputs={len(segs)}:normalize=0,apad=whole_dur=40,atrim=0:40,"
         "highpass=f=80,acompressor=threshold=-20dB:ratio=3:attack=5:release=120,volume=1.6,asplit[vo][vokey]")
f.append("[0:a]volume=0.8[mus];[mus][vokey]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[duck]")
f.append("[duck][vo]amix=inputs=2:normalize=0,alimiter=limit=0.95,atrim=0:40[outa]")
cmd=["ffmpeg","-y","-loglevel","error","-i","score.wav","-i",f"vo/{lang}.mp3","-filter_complex",";".join(f),"-map","[outa]","-ar","48000",out]
subprocess.run(cmd,check=True)
