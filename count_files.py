import os
import pickle
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    
    if len(args) != 1:
        print ('FORMAT: python3 count_files.py directory_name')
        quit()
        
    direc = args[0]

    events = pickle.load(open('{}/actions-info-dict.pkl'.format(direc), 'rb'))

    # events['steal success'].append('9aOeOCLRYOM/clip_6.mp4')
    # events['layup success'].append('vwHTKAPgaqA/clip_10.mp4')
    # events['3-pointer failure'].append('IIF1TFwBe2A/clip_5.mp4')
    # events['layup success'].append('61st5kS66oE/clip_7.mp4')
    # events['other 2-pointer failure'].append('Gqfq4AaUJe0/clip_15.mp4')
    # events['other 2-pointer success'].append('Gqfq4AaUJe0/clip_16.mp4')

    # # final
    # with open('{}/actions-info-dict.pkl'.format(direc), 'wb') as handle:
    #     pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_clips = set()
    errors = []
    for action, clips in events.items():
        # print (action, clips)
        print (len(list(set(clips))) == len(clips))
        print (action, '\n')
        for clip in clips:
            if clip in all_clips:
                 errors.append(clip)

            all_clips.add(clip)
            print (clip)

        print ('\n\n')

    print ('Num Files in Pkl:', len(all_clips))

    count = 0
    for chunk in os.listdir('{}/video-data'.format(direc)):    
        if chunk[0] == '.':
            continue

        for file in os.listdir('{}/video-data/{}'.format(direc, chunk)):    
            count += 1
    print ('Num Files:', count)

    print ('Errors:')
    for error in errors:
        print (error)

    # 9aOeOCLRYOM/clip_6.mp4 steal success
    # vwHTKAPgaqA/clip_10.mp4 layup success
    # IIF1TFwBe2A/clip_5.mp4 3-pointer failure
    # 61st5kS66oE/clip_7.mp4 layup success
    # Gqfq4AaUJe0/clip_15.mp4 other 2-pointer failure
    # Gqfq4AaUJe0/clip_16.mp4 other 2-pointer success
