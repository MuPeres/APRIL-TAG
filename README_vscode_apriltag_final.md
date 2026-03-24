# AprilTag no VSCode — pacote final

## Arquivos
- `apriltag_common_final.py`
- `calibrate_from_video_final.py`
- `live_pose_from_screen_final.py`
- `requirements_vscode_apriltag_final.txt`

## Instalação
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_vscode_apriltag_final.txt
```

## Calibração
Exemplo com os dois vídeos mais recentes:
```powershell
python calibrate_from_video_final_seed_estavel.py --video "C:\Projetos\apriltag_live\L50_02.mp4" "C:\Projetos\apriltag_live\L50_01.mp4" --out-dir "C:\Projetos\apriltag_live\output_calib" --tag-id 35 --tag-size-mm 28 --fix-k2 --fix-k3 --min-sharp-abs 10 --min-tag-width-px 240 --seed-views 20 --seed-min-sharp 15 --seed-min-width-px 240
```

Saída esperada:
- `output_calib\calib_intrinsics_apriltag.npz`
- `output_calib\calib_intrinsics_apriltag.json`

## Seleção da ROI do preview
Abra o preview da câmera no navegador e rode:
```powershell
python live_pose_from_screen_final_pitch_offset.py --calib-npz "C:\Projetos\apriltag_live\output_calib\calib_intrinsics_apriltag.npz" --select-roi
```

## Live pose
```powershell
python live_pose_from_screen_final_pitch_offset.py --calib-npz "C:\Projetos\apriltag_live\output_calib\calib_intrinsics_apriltag.npz" --tag-id 35 --tag-size-m 0.028 --image-mode resize --min-tag-width-px 20
```

## Atalhos no live
- `q` ou `Esc`: sair
- `r`: resetar âncora
- `s`: salvar snapshot
- `p`: imprimir pose atual no terminal

