python fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt "A person is drizzling olive oil over a pizza with a spoon from a glass jar. The pizza has a golden-brown crust topped with dollops of white cheese and red sauce. The background shows a wooden table surface, indicating the setting is likely a kitchen. There is no significant movement in the scene, focusing solely on the action of oil being poured. The lighting is bright, enhancing the colors of the food. The scene continues with the person pouring olive oil on the pizza, focusing on the central area. The amount of oil on the pizza visibly increases, giving it a shiny appearance. The spoon used for drizzling is partially visible, held by a hand entering from the upper part of the frame. The consistency of the oil is clear and fluid. The background remains unchanged, maintaining the focus on the pizza. The oil pouring action concludes, and the spoon is being withdrawn from the frame, dripping oil back into the jar. The pizza now looks glossier due to the freshly added oil, emphasizing the texture of the cheese and sauce. The hand and spoon move away, indicating the end of this specific food preparation step. The wooden table and the kitchen setting remain constant throughout. The scene ends with the pizza fully dressed and more visually appealing due to the added oil." \
    --num_frames 79 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --seed 12346 \
    --output_path outputs4.mp4