from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.drift_detection import DDM

# 1. Create a stream
ddm = DDM()
stream = FileStream("olist_prepared.csv", target_idx=13)
stream.prepare_for_use()

file = open("olist_stream_results.txt","w");

# 2. Instantiate the HoeffdingTree classifier
ht = HoeffdingTree()

# 3. Setup the evaluator
evaluator = EvaluatePrequential(n_wait=30, pretrain_size=1, max_samples=10000, metrics=['accuracy','true_vs_predicted'], show_plot=True, restart_stream=True)

# 4. Run evaluation
evaluator.evaluate(stream=stream, model=ht)

X, y_true = stream.next_sample()
ht.partial_fit(X,y_true)

file.write('DRIFT DETECTIONS\n')
drifts = 0
for i in range(10000):
     y_pred = ht.predict(X)
     ddm.add_element(y_true == y_pred)
     X, y_true = stream.next_sample()
     ht.partial_fit(X, y_true)
     if ddm.detected_warning_zone():
         file.write('Warning zone has been detected in data of index: ' + str(i) + '\n')
     if ddm.detected_change():
         file.write('Change has been detected in data of index: ' + str(i) + '\n')
         drifts = drifts + 1

file.write('\n%d DRIFTS DETECTED.' % (drifts))
file.close()