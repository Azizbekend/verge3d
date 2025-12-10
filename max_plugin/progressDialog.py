from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

from pymxs import runtime as rt

class ProgressDialog:

    @classmethod
    def open(cls, title=''):
        rt.execute('''
            try(destroyDialog progDial) catch()
            rollout progDial "{}" (
                progressbar bar color:[95, 138, 193] pos: [50, 15] width: 300 height: 20
            )

            createDialog progDial 400 50 style: #(#style_titlebar, #style_sysmenu)
        '''.format(title))
        cls.setValue(0)

    @classmethod
    def close(cls):
        rt.execute('try(destroyDialog progDial) catch()')

    @classmethod
    def setValue(cls, value):
        rt.execute('try(progDial.bar.value = {}) catch()'.format(str(value)))
