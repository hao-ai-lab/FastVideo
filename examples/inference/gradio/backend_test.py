import argparse
import os
import time
import json
import statistics
import asyncio
import aiohttp
from copy import deepcopy
from typing import List, Dict, Any
import threading

import torch

# All the prompts for stress testing
STRESS_TEST_PROMPTS = [
    "A person reading a book with words that float off the pages and form pictures.",
    "A person diving into a pool of liquid crystal, creating ripples of light.",
    "A handheld shot chasing after a group of friends laughing and playing on the beach at sunset.",
    "A mysterious ancient temple hidden in the jungle.",
    "A high-speed train navigating a steep descent.",
    "a toy robot wearing blue jeans and a white t shirt taking a pleasant stroll in Antarctica during a winter storm",
    "A cheetah accelerating to full speed while chasing its prey.",
    "A serene orchard is in full bloom, with trees heavy with blossoms and bees buzzing around, darting from flower to flower in a display of natural harmony.",
    "A little child let out a big yawn",
    "Subtle reflections of a woman on the window of a train moving at hyper-speed in a Japanese city.",
    "A truck left along the edge of a cliff, revealing the stunning coastal landscape below with waves crashing against the rocks.",
    "A red bird transforms into a flag",
    "A zoom-out from a single leaf on a tree to reveal the entire forest, showcasing the vastness and diversity of the woodland.",
    "A slow-motion video of a liquid droplet bouncing on a water-repellent surface.",
    "Static camera shot. A dinasour running near some lions and chasing them away.",
    "an adorable kangaroo wearing purple overalls and cowboy boots taking a pleasant stroll in Mumbai India during a beautiful sunset",
    "A zoom-in on an artist's brush touching the canvas, highlighting the texture of the paint and the strokes being made.",
    "an old man wearing blue jeans and a white t shirt taking a pleasant stroll in Mumbai India during a colorful festival",
    "A woman is ascending to the sky from the ground",
    "View out a window of a giant strange creature walking in rundown city at night, one single street lamp dimly lighting the area.",
    "An arc shot around a lone tree in a vast, foggy field at dawn, revealing the changing light and shadows.",
    "A person sculpting a statue out of a waterfall, the water solidifying under their touch.",
    "The person's forehead creased with concentration as she worked on a challenging puzzle.",
    "The person's cheeks flushed with pleasure as she savored a delicious meal.",
    "Hand-drawn simple line art, a young kid looking up into space with a wondrous expression on his face.",
    "A crab made of different jewlery is walking on the beach. As it walks, it drops different jewelry pieces like diamonds, pearls, etc",
    "Gold coins are falling out when elevator door opens",
    "the scene transitions from huge waves into a snowy mountain at sunset",
    "a giant cathedral is completely filled with cats. there are cats everywhere you look. a man enters the cathedral and bows before the giant cat king sitting on a throne.",
    "A mother dog gently picks up a piece of meat and carefully places it in her puppy's bowl, her eyes filled with warmth and care as she watches her little one eat.",
    "A soap bubble floating in the air, displaying iridescent colors that shift and change as it moves through different angles of light.",
    "A truck left alongside a train moving through the countryside, matching its speed and revealing the changing landscape.",
    "An astronaut walking between stone buildings.",
    "A close-up shot of the person's face reveals his fear and desperation as he navigates the ship through the storm.",
    "A frozen lake slowly cracking and thawing as spring arrives, with sheets of ice breaking apart and drifting across the surface.",
    "A FPV shot zooming through a tunnel into a vibrant underwater space.",
    "a toy robot wearing blue jeans and a white t shirt taking a pleasant stroll in Mumbai India during a colorful festival",
    "A person sips on a smoothie, the cool and fruity flavors refreshing her mouth.",
    "In a vibrant theater, a magician in dazzling attire stands center stage, pulling a comically oversized rubber chicken from an ornate, old-fashioned box. His costume shimmers under the stage lights, adding to the spectacle. The crowd erupts in laughter and applause, their faces filled with joy and amazement. The magician's expression hints at mischievous delight as he holds up the rubber chicken, his performance bringing cheer to the audience.",
    "A hamster running on a spinning wheel.",
    "A quaint village nestled in a valley is surrounded by blooming cherry blossoms, with petals drifting through the air as villagers go about their daily activities, adding life to the scene.",
    "In a tranquil forest clearing, a sparkling waterfall cascades down into a clear pool, surrounded by lush greenery and flowers, with occasional birds fluttering by.",
    "A woman beamed with pride as she watched her child perform on stage.",
    "an adorable kangaroo wearing blue jeans and a white t shirt taking a pleasant stroll in Mumbai India during a winter storm",
    "A man is eating salad",
    "An Asian girl wearing a bright yellow T-shirt and white pants is Hip-Hop dancing",
    "nighttime footage of a hermit crab using an incandescent lightbulb as its shell",
    "a toy robot wearing a green dress and a sun hat taking a pleasant stroll in Antarctica during a beautiful sunset",
    "A goat operating a food truck, serving gourmet grilled cheese sandwiches to a line of animals.",
    "Macro shot. Man in an antique scuba helmet with dark glass walking out of a flower",
    "A bustling train station in the heart of a vibrant city.",
    "Light filtering through a canopy of autumn leaves, casting warm, dappled patterns of yellow, orange, and red onto the ground.",
    "Chimneys in the setting sun",
    "A longboarder accelerating downhill, carving through turns.",
    "A couple runs through a sudden downpour, laughing and splashing in puddles as they try to find shelter.",
    "A glass of iced coffee condensing water on the outside, with droplets forming and sliding down the glass in slow motion.",
    "macro shot of a leaf showing tiny trains moving through its veins",
    "A corgi wearing sunglasses walks on the beach of a tropical island",
    "Borneo wildlife on the Kinabatangan River",
    "A beautiful silhouette animation shows a wolf howling at the moon, feeling lonely, until it finds its pack.",
    "an adorable kangaroo wearing blue jeans and a white t shirt taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "A green monster made of plants walks through an airport.",
    "A close up view of a glass sphere that has a zen garden within it. There is a small dwarf in the sphere who is raking the zen garden and creating patterns in the sand.",
    "A person on a scooter colliding with a park bench, the scooter tipping over.",
    "A tilt-up from a city street, ascending to show the skyline with its mix of modern and historic architecture.",
    "A chef tossing a pancake into the air and catching it.",
    "A woman whispering a secret into a friend's ear.",
    "A vulture circling high in the sky.",
    "A medieval castle overlooking a bustling renaissance fair.",
    "a toy robot wearing purple overalls and cowboy boots taking a pleasant stroll in Mumbai India during a beautiful sunset",
    "A man standing in front of a burning building giving the 'thumbs up' sign.",
    "The person's cheeks flushed with embarrassment as he told a funny story.",
    "Llamas and Emus are playing chess",
    "A woman sipping a steaming cup of tea.",
    "A tree root bursting through the seat of an ancient, weathered bench, intertwining with the wood.",
    "Smoke rises from the chimney of a cozy log cabin nestled in the woods, with soft light glowing from the windows, suggesting a warm and inviting atmosphere.",
    "A close-up of sparkling water being poured into a glass, capturing the detailed flow and bubbles.",
    "a woman wearing blue jeans and a white t shirt taking a pleasant stroll in Antarctica during a beautiful sunset",
    "The Glenfinnan Viaduct is a historic railway bridge in Scotland, UK, that crosses over the west highland line between the towns of Mallaig and Fort William. It is a stunning sight as a steam train leaves the bridge, traveling over the arch-covered viaduct. The landscape is dotted with lush greenery and rocky mountains, creating a picturesque backdrop for the train journey. The sky is blue and the sun is shining, making for a beautiful day to explore this majestic spot.",
    "A piece of elastic fabric being pulled and stretched, then returning to its original size when the tension is released.",
    "a woman wearing a green dress and a sun hat taking a pleasant stroll in Antarctica during a beautiful sunset",
    "A video of a water jet cutting through metal, showing the powerful and precise movement of water.",
    "Car mirrors and sunsets",
    "Giant Pandas are eating hot noodles in a Chinese restaurant",
    "A rally car taking a fast turn on a track",
    "a toy robot wearing purple overalls and cowboy boots taking a pleasant stroll in Mumbai India during a colorful festival",
    "A crystal-clear icicle slowly dripping as it melts in the warmth of the midday sun, each drop sparkling as it falls.",
    "A tilt-down from a chandelier in a grand hall, revealing the ornate decor and people mingling below.",
    "A man is playing the drums under the water",
    "A person playing an electric guitar made of lightning, with thunderous sound waves.",
    "A person floating in a bubble, drifting over a bustling cityscape.",
    "A tilt-down from a starry night sky, revealing a quiet forest clearing bathed in moonlight.",
    "A pan right through a dense jungle, moving past lush vegetation and exotic wildlife.",
    "Close-up of a man eating an apple.",
    "A low-angle shot of a dancer leaping gracefully into the air, making their movement appear even more dynamic and powerful.",
    "A woman is search her bag trying to find something.",
    "A bulldozer clears debris from a demolished building, making way for new construction.",
    "A man sighed in relief as the doctor delivered the good news.",
    "A tsunami coming through an alley in Bulgaria, dynamic movement.",
    "Blooming Flowers",
    "A push-in through a dense crowd at a festival, moving towards a performer on stage who is captivating the audience.",
    "A truck right through a tranquil garden, moving past blooming flowers, trees, and a small fountain.",
    "The person's eyes sparkled with excitement as he greeted a friend.",
    "A person playing chess with a robot on a floating platform above the ocean.",
    "A gentle breeze rustles the leaves as someone walks down a serene forest path, sunlight filtering through the trees and shifting patterns on the ground as branches sway.",
    "A rollercoaster ride from a city to a desert and then to an ice world",
    "A pan left across an ancient library, moving from shelf to shelf, showcasing rows of leather-bound books.",
    "A mother otter floating on her back in a river, cradling her pup on her stomach to keep it safe and warm in the gentle current.",
    "an adorable kangaroo wearing purple overalls and cowboy boots taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "a woman wearing a green dress and a sun hat taking a pleasant stroll in Mumbai India during a colorful festival",
    "A delicate layer of morning frost melting off a flower petal, the tiny droplets glistening like diamonds in the light.",
    "A panda is cooking for her child, her child is next to her.",
    "Macro shot of a man wearing an antique diving helmet with dark glass and a jetpack walking on the veins of a leaf. Realistic style",
    "an old man wearing purple overalls and cowboy boots taking a pleasant stroll in Johannesburg South Africa during a beautiful sunset",
    "A girl is unfolding a birthday gift.",
    "A pencil drawing an architectural plan.",
    "A handheld camera following a dog running through a park, bouncing and tilting as it captures the dog's joyful exploration.",
    "A pan left across a serene beach at sunrise, moving from the darkened shore to the brightening horizon.",
    "A group of people are clapping to celebrate",
    "Vendors set up stalls at a bustling farmer's market, displaying fresh fruits and vegetables, while people stroll through, selecting produce and enjoying the lively atmosphere.",
    "A police helicopter hovers above a high-speed chase, guiding officers on the ground to apprehend a suspect.",
    "A paper origami dragon riding a boat in waves. Realistic style.",
    "A close-up of a droplet of dew forming on a leaf, capturing the detailed surface tension.",
    "a toy robot wearing blue jeans and a white t shirt taking a pleasant stroll in Mumbai India during a beautiful sunset",
    "A dry rainbow rose is coming back to life.",
    "A glass falling off a table and shattering on the floor.",
    "A marathon runner crossing the finish line after a grueling race.",
    "A zoom-in on a drop of morning dew on a leaf, showing the reflection of the surrounding world within it.",
    "A child blowing on hot cocoa to cool it down.",
    "A squad of futsal players showcasing their skills on an indoor court.",
    "A princess is brushing her long golden hair in the garden.",
    "A close-up of a pair of eyes, revealing the subtle emotions and reflections within them.",
    "A tracking shot of a group of cyclists racing through a forest trail, with trees and foliage rushing by.",
    "A woman yawning widely at the end of a long day.",
    "an old man wearing a green dress and a sun hat taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "Hidden within a garden, an ancient fountain trickles with water, surrounded by vibrant flowers and lush greenery that seem to whisper secrets of the past.",
    "A Chinese man sits at a table and eats noodles with chopsticks",
    "A pink pig running fast toward the camera in an alley in Tokyo.",
    "Strange creatures move through a mysterious, foggy marsh, their silhouettes barely visible through the dense mist as they navigate the eerie, otherworldly landscape.",
    "Tour of an art gallery with many beautiful works of art in different styles.",
    "FPV flying through a colorful coral lined streets of an underwater suburban neighborhood.",
    "Aerial view of Santorini during the blue hour, showcasing the stunning architecture of white Cycladic buildings with blue domes. The caldera views are breathtaking, and the lighting creates a beautiful, serene atmosphere.",
    "Camera zoom out. A couple walking along the beach as the sun sets over the ocean.",
    "an extreme close up shot of a woman's eye, with her iris appearing as earth",
    "a woman wearing purple overalls and cowboy boots taking a pleasant stroll in Mumbai India during a colorful festival",
    "an old man wearing a green dress and a sun hat taking a pleasant stroll in Mumbai India during a winter storm",
    "an adorable kangaroo wearing blue jeans and a white t shirt taking a pleasant stroll in Antarctica during a winter storm",
    "A martial artist breaking a board with a powerful punch.",
    "People gather on a peaceful beach at sunset, a bonfire crackling as they sit around, enjoying the warmth and the sight of the sun dipping below the horizon.",
    "A close-up of a waterfall, showing the detailed movement of water as it crashes down.",
    "A child is blowing bubbles",
    "a woman wearing a green dress and a sun hat taking a pleasant stroll in Johannesburg South Africa during a winter storm",
    "A wide-angle perspective of a serene lake surrounded by mountains, reflecting the sky and creating a sense of infinite space.",
    "The person's eyebrows arched in skepticism as she listened to a dubious claim.",
    "an old man wearing blue jeans and a white t shirt taking a pleasant stroll in Mumbai India during a beautiful sunset",
    "a woman wearing a green dress and a sun hat taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "a woman wearing purple overalls and cowboy boots taking a pleasant stroll in Antarctica during a colorful festival",
    "a toy robot wearing blue jeans and a white t shirt taking a pleasant stroll in Antarctica during a colorful festival",
    "A chef flips a pancake and puts cream on it.",
    "An astronaut runs on the surface of the moon, the low angle shot shows the vast background of the moon, the movement is smooth and appears lightweight",
    "A man's face lit up with happiness as he received a heartfelt compliment.",
    "A futuristic spaceport hums with activity as ships of various shapes and sizes take off and land on multiple platforms, their engines glowing with vibrant colors.",
    "A person knitting a scarf using beams of light instead of yarn.",
    "A pedestal up from the edge of a canyon, gradually revealing the expansive landscape and river below.",
    "a woman wearing purple overalls and cowboy boots taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "an old man wearing blue jeans and a white t shirt taking a pleasant stroll in Johannesburg South Africa during a colorful festival",
    "A person walking up a staircase made of clouds leading to a floating castle.",
    "Monks meditate in a serene mountaintop temple, sitting in quiet reflection as the wind gently moves through the surrounding trees, creating a sense of peace and tranquility.",
    "An aerial shot of a bustling city intersection at rush hour, capturing the organized chaos of cars and pedestrians.",
    "A pair of hands skillfully knitting a colorful scarf, the yarn winding through their fingers with each stitch.",
    "Close-up, a Chinese child is eating dumplings",
    "A kite losing wind and falling to the ground.",
    "Bioluminescent waves gently wash ashore on a deserted beach, illuminating the sand with each cresting wave as a figure walks along the water's edge, leaving glowing footprints.",
    "A red panda taking a bite of a pizza",
    "A close-up shot of a young woman driving a car, looking thoughtful, blurred green forest visible through the rainy car window.",
    "A high-speed video of a splash created by a stone thrown into a pond.",
    "A metal rod being bent slightly by a force and then springing back to its original straight shape when the force is removed.",
    "A hedgehog in a knight's armor, riding a toy horse into a medieval castle.",
    "A bird made of fresh oranges rushes out of the orange",
    "A low altitude first person perspective camera tracking shot of a soccer player's feet dribbling the ball on the groud in a soccer field, Sports Videography, Motion Tracking camera shot",
    "A tranquil island retreat features swaying palm trees and hammocks strung between them, inviting guests to relax and enjoy the serene beauty of the surroundings.",
    "a spooky haunted mansion, with friendly jack o lanterns and ghost characters welcoming trick or treaters to the entrance, tilt shift photography",
    "A coconut tree made of dollar bills at sunset, with bills falling off like leaves.",
    "A motocross bike accelerating out of a tight turn on a dirt track.",
    "A tranquil Zen garden with a gently flowing stream and koi fish.",
    "A green monster made of leaves walks through the airport, carrying a suitcase.",
    "A time-lapse of a frost-covered leaf gradually thawing in the morning sunlight, with tiny water droplets forming and trickling down.",
    "A woman practicing her archery skills at a range.",
    "A slow-motion video of ink being injected into a tank of water, creating intricate and beautiful patterns.",
    "a woman wearing blue jeans and a white t shirt taking a pleasant stroll in Johannesburg South Africa during a winter storm",
    "The person's forehead creased with worry as he listened to bad news.",
    "An arc shot around a grand piano being played in an empty concert hall, the motion revealing the intricate details of the instrument.",
    "A person conducting a symphony of animals in a forest clearing.",
    "A truck right alongside a flowing river, capturing the movement of the water and the surrounding forest.",
    "A rocket blasting off from the launch pad, accelerating rapidly into the sky.",
    "Workers move through a picturesque vineyard during the harvest season, carefully picking grapes and placing them into baskets as the sun bathes the vines in a warm glow.",
    "A person is eating an ice cream.",
    "An over-the-shoulder perspective of a chef meticulously plating a dish in a bustling kitchen.",
    "A man looked away in shame when confronted with his wrongdoing.",
    "A person is savoring a slice of pizza at a pizzeria."
]

class BackendStressTest:
    def __init__(self, output_path: str, 
                 server_url: str = "http://localhost:8000", max_concurrent: int = 50):
        self.output_path = output_path
        self.server_url = server_url
        self.max_concurrent = max_concurrent
        
        # Results storage
        self.results = []
        self.lock = threading.Lock()
        
    async def check_health(self) -> bool:
        """Check if the Ray Serve backend is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
        
    def build_request_params(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build request parameters for Ray Serve backend"""
        # Default parameters matching the Ray Serve backend
        default_params = {
            'prompt': prompt,
            'negative_prompt': None,
            'use_negative_prompt': False,
            'seed': 42,
            'guidance_scale': 7.5,
            'num_frames': 21,
            'height': 448,
            'width': 832,
            'num_inference_steps': 20,
            'randomize_seed': True,
            'return_frames': False  # Don't return frames for stress testing to reduce overhead
        }
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if key in default_params:
                default_params[key] = value
        
        # Randomize seed if requested
        if default_params.get('randomize_seed', True):
            default_params['seed'] = torch.randint(0, 1000000, (1,)).item()
            
        # Handle negative prompt
        if not default_params.get('use_negative_prompt', False):
            default_params['negative_prompt'] = None
        
        # NEW: Remove keys with None values to avoid sending nulls that may break validation
        clean_params = {k: v for k, v in default_params.items() if v is not None}
        return clean_params
    
    async def test_single_request(self, session: aiohttp.ClientSession, prompt: str, request_id: int) -> Dict[str, Any]:
        """Test a single request and measure latency"""
        start_time = time.time()
        
        try:
            # Build request parameters
            request_params = self.build_request_params(prompt)
            
            # Make request to Ray Serve backend
            async with session.post(
                f"{self.server_url}/generate_video",
                json=request_params,
                timeout=aiohttp.ClientTimeout(total=900)  # 15 minute timeout for video generation
            ) as response:
                
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    if response_data.get('success', False):
                        result = {
                            'request_id': request_id,
                            'prompt': prompt,
                            'latency': latency,
                            'status': 'success',
                            'response_time': latency,  # Use our own timing
                            'timestamp': start_time,
                            'output_path': response_data.get('output_path', ''),
                            'used_seed': response_data.get('seed', request_params['seed'])
                        }
                    else:
                        result = {
                            'request_id': request_id,
                            'prompt': prompt,
                            'latency': latency,
                            'status': 'error',
                            'error': response_data.get('error_message', 'Unknown backend error'),
                            'timestamp': start_time
                        }
                else:
                    response_text = await response.text()
                    result = {
                        'request_id': request_id,
                        'prompt': prompt,
                        'latency': latency,
                        'status': 'error',
                        'error': f"HTTP {response.status}: {response_text}",
                        'timestamp': start_time
                    }
                    
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            result = {
                'request_id': request_id,
                'prompt': prompt,
                'latency': latency,
                'status': 'error',
                'error': str(e),
                'timestamp': start_time
            }
        
        # Thread-safe result storage
        with self.lock:
            self.results.append(result)
            
        return result
    
    async def run_stress_test(self, num_iterations: int = 1, concurrent_requests: int = None):
        """Run the stress test with multiple iterations and concurrent requests"""
        if concurrent_requests is None:
            concurrent_requests = self.max_concurrent
            
        # Check backend health before starting
        print(f"Testing Ray Serve backend at {self.server_url}...")
        if not await self.check_health():
            print(f"❌ Backend is not healthy at {self.server_url}")
            print("Make sure the Ray Serve backend is running with:")
            print("python ray_serve_backend.py")
            return
        print("✅ Backend is healthy and ready for stress testing")
        
        print(f"\nStarting stress test with {len(STRESS_TEST_PROMPTS)} prompts")
        print(f"Running {num_iterations} iteration(s) with {concurrent_requests} concurrent requests")
        print(f"Total requests: {len(STRESS_TEST_PROMPTS) * num_iterations}")
        print(f"Backend URL: {self.server_url}")
        print("-" * 80)
        
        all_prompts = STRESS_TEST_PROMPTS * num_iterations
        request_id = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def limited_request(session: aiohttp.ClientSession, prompt: str, req_id: int):
            async with semaphore:
                return await self.test_single_request(session, prompt, req_id)
        
        # Run concurrent requests using asyncio
        async with aiohttp.ClientSession() as session:
            # Create all tasks
            tasks = [
                limited_request(session, prompt, request_id + i)
                for i, prompt in enumerate(all_prompts)
            ]
            
            # Process completed requests as they finish
            completed = 0
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    completed += 1
                    prompt = result['prompt']
                    status_icon = "✅" if result['status'] == 'success' else "❌"
                    output_info = f" -> {result.get('output_path', 'N/A')}" if result['status'] == 'success' else ""
                    print(f"{status_icon} [{completed}/{len(all_prompts)}] {result['latency']:.2f}s - {prompt[:50]}...{output_info}")
                except Exception as e:
                    completed += 1
                    print(f"❌ [{completed}/{len(all_prompts)}] Exception: {e}")
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and print test results"""
        print("\n" + "=" * 80)
        print("STRESS TEST RESULTS")
        print("=" * 80)
        
        successful_requests = [r for r in self.results if r['status'] == 'success']
        failed_requests = [r for r in self.results if r['status'] == 'error']
        
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful: {len(successful_requests)}")
        print(f"Failed: {len(failed_requests)}")
        print(f"Success Rate: {len(successful_requests)/len(self.results)*100:.1f}%")
        
        if successful_requests:
            latencies = [r['latency'] for r in successful_requests]
            print(f"\nLatency Statistics (seconds):")
            print(f"  Min: {min(latencies):.2f}")
            print(f"  Max: {max(latencies):.2f}")
            print(f"  Mean: {statistics.mean(latencies):.2f}")
            print(f"  Median: {statistics.median(latencies):.2f}")
            print(f"  Std Dev: {statistics.stdev(latencies):.2f}")
            
            # Percentiles
            sorted_latencies = sorted(latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
            p90 = sorted_latencies[int(len(sorted_latencies) * 0.9)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            print(f"  P50: {p50:.2f}")
            print(f"  P90: {p90:.2f}")
            print(f"  P95: {p95:.2f}")
            print(f"  P99: {p99:.2f}")
        
        if failed_requests:
            print(f"\nFailed Requests ({len(failed_requests)}):")
            for req in failed_requests[:5]:  # Show first 5 failures
                print(f"  - {req['error']}")
            if len(failed_requests) > 5:
                print(f"  ... and {len(failed_requests) - 5} more")
        
        # Save detailed results
        results_file = os.path.join(self.output_path, "stress_test_results.json")
        os.makedirs(self.output_path, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_requests': len(self.results),
                    'successful_requests': len(successful_requests),
                    'failed_requests': len(failed_requests),
                    'success_rate': len(successful_requests)/len(self.results)*100 if self.results else 0
                },
                'latency_stats': {
                    'min': min(latencies) if successful_requests else 0,
                    'max': max(latencies) if successful_requests else 0,
                    'mean': statistics.mean(latencies) if successful_requests else 0,
                    'median': statistics.median(latencies) if successful_requests else 0,
                    'std_dev': statistics.stdev(latencies) if len(successful_requests) > 1 else 0
                },
                'detailed_results': self.results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")

async def main():
    parser = argparse.ArgumentParser(description="FastVideo Ray Serve Backend Stress Test")
    parser.add_argument("--output_path",
                        type=str,
                        default="outputs",
                        help="Path to save test results")
    parser.add_argument("--server_url",
                        type=str,
                        default="http://localhost:8000",
                        help="Ray Serve backend URL")
    parser.add_argument("--max_concurrent",
                        type=int,
                        default=50,
                        help="Maximum concurrent requests")
    parser.add_argument("--iterations",
                        type=int,
                        default=1,
                        help="Number of iterations through all prompts")
    parser.add_argument("--concurrent_requests",
                        type=int,
                        default=None,
                        help="Number of concurrent requests (overrides max_concurrent)")
    
    args = parser.parse_args()
    
    # Create stress test instance
    stress_test = BackendStressTest(
        output_path=args.output_path,
        server_url=args.server_url,
        max_concurrent=args.max_concurrent
    )
    
    # Run the stress test
    await stress_test.run_stress_test(
        num_iterations=args.iterations,
        concurrent_requests=args.concurrent_requests
    )


if __name__ == "__main__":
    asyncio.run(main())
