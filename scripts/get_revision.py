# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:01:55 2013

@author: Jan
"""

def make_hgrevision(target, source, env):
    import subprocess as sp
    output = sp.Popen(["hg", "id", "-i"], stdout=sp.PIPE).communicate()[0]

    hgrevision_cc = file(str(target[0]), "w")
    hgrevision_cc.write('HG_Revision = "{0}"\n'.format(output.strip()))
    hgrevision_cc.close()