I want to allow multiple TRILEGAL downloads simultaneously.  I beleive this would require separate folders.  This is almost set up in vespa, but it needs to carry through to the perl script.  I will modify those here, and point to these specifically when running the new code.

```
$ cp ~/anaconda/envs/py36/bin/get_trilegal .
$ cp ~/anaconda/envs/py36/lib/python3.6/site-packages/vespa/stars/trilegal.py .
```

Within trilegal.py:
* I added one more argument to be sent to get_trilegal = the folder name

Within get_trilegal:
* I prepended that folder name to lixo and tmpfile and most of the filename

Within extinction:
* I added a check so that it doesn't crash if the tmp file does not exist (happens sometimes in MPI; maybe different processes trying to write/read to/from the same file?)
