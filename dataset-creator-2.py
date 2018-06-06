import subprocess
import os
import queue
import threading
import pickle

## CHANGE BOTH
RUNNING = 'andrew-vm-4'
SKIP = 3000 # INCLUSIVE begin
END = 4000 # EXCLUSIVE end


def convert(milliseconds):
    seconds = milliseconds / 1e3
    minutes = seconds // 60
    hours = minutes // 60

    return "%02d:%02d:%02d.%03d" % (hours, minutes % 60, seconds % 60, milliseconds % 1000)


def download_link(line):
    global row
    with row_lock:
        row += 1
        if row % 20 == 0:
            print ('At Row {}'.format(row))

    (vid_id, width, height, clip_start, clip_end, event_start, event_end, ball_x, ball_y, label, type) = line.strip().split(',')

    newname = None
    fullname = None
    with counts_lock:
        if vid_id not in counts:
            counts[vid_id] = 0

        counts[vid_id] += 1

        newname = '{}/clip_{}.mp4'.format(vid_id, counts[vid_id])
        fullname = '{}/video-data/{}'.format(RUNNING, newname)

    url = 'https://www.youtube.com/watch?v={}'.format(vid_id)

    event_end = float(event_end)
    start = convert(event_end - 2000)

    if not os.path.exists(os.path.dirname(fullname)):
        try:
            os.makedirs(os.path.dirname(fullname))
            # rewrite
        except: # Guard against race condition
            # add back to queue
            with q_lock:
                q.put(line)

            # exit
            return
          
    try:
        # take 4 second clip 6 fps
        subprocess.check_output('ffmpeg -y -i $(youtube-dl -f "best[height=360,width=490]" --get-url {}) -s 160x160 \
            -ss {} -t 00:00:03.000 -vf fps="fps=6" -an -hide_banner -loglevel panic\
            {}'.format(url, start, fullname), shell=True)
    except: # Guard against race condition
        # add back to queue
        # q_lock.acquire()
        # q.put(line)
        # q_lock.release()

        # exit
        return          

    # lock output dict
    with output_lock:

        if label not in output_dict:
            output_dict[label] = []

        output_dict[label].append(newname)

        # lock count
        with overall_count_lock:   
            # needed to write to overall_count
            global overall_count

            overall_count += 1
            if overall_count % 20 == 0:
                print ('{} videos downloaded'.format(overall_count))

                with open('{}/actions-info-dict.pkl'.format(RUNNING), 'wb') as handle:
                    pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            



# called by each thread
def download_helper(i):
    while True:
        try:
            with q_lock:
                # check if empty
                if q.empty():
                    break

                line = q.get()

            download_link(line)
        except:
            # release locks
            continue

    print ('thread {} finished!'.format(i))


q = queue.Queue()
q_lock = threading.Lock()

counts = {}
counts_lock = threading.Lock()

output_dict = {}
output_lock = threading.Lock()

row = SKIP
row_lock = threading.Lock()

overall_count = 0
overall_count_lock = threading.Lock()

# 52 + 31 errored out
with open('bball_dataset_april_4.csv', 'r') as datafile:
    count = 0
    for line in datafile:
        if count >= END:
            break
            
        if count >= SKIP:
            q.put(line)

        count += 1

# start threads
num_threads = 5
threads = []
for i in range(num_threads):
    threads.append(threading.Thread(target=download_helper, args=(i,)))
    threads[i].daemon = True
    threads[i].start()

# wait for finish
for i in range(num_threads):
    threads[i].join()

# final
with open('{}/actions-info-dict.pkl'.format(RUNNING), 'wb') as handle:
    pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
