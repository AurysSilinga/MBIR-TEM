import os
import distutils.sysconfig
import sys

PYTHON_LIBPATH = distutils.sysconfig.get_python_lib()
PYTHON_INCPATH = distutils.sysconfig.get_python_inc()
PYTHON_LIBRARY = "python" + sys.version[:3]

print 'PYTHON_LIBPATH: ' + PYTHON_LIBPATH
print 'PYTHON_INCPATH: ' + PYTHON_INCPATH
print 'PYTHON_LIBRARY: ' + PYTHON_LIBRARY

env = Environment(ENV = os.environ)

if ARGUMENTS.get('VERBOSE') != '1':
    env['CCCOMSTR'] = "Compiling $TARGET"
    env['LINKCOMSTR'] = "Linking $TARGET"

Progress('$TARGET\r', overwrite=True)

env.AppendUnique(LIBPATH=[PYTHON_LIBPATH], CPPPATH=[PYTHON_INCPATH])

env.Program('hello', 'hellotest.c', CPPPATH='.')

env.Decider('MD5-timestamp')