import time
import sys

class WatchdogTimeoutError(RuntimeError):
	"""Raised in case of runtime limit violations."""
	pass

def watchdog(timeout, code, *args, **kwargs):
	# "Time-limited execution."
	def tracer(frame, event, arg, start=time.time()):
		# "Helper."
		now = time.time()
		if now > start + timeout:
			raise WatchdogTimeoutError(start, now)
		return tracer if event == "call" else None

	old_tracer = sys.gettrace()
	try:
		sys.settrace(tracer)
		return code(*args, **kwargs)
	finally:
		sys.settrace(old_tracer)