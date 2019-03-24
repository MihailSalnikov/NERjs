const MAX_SEQUENCE_LENGTH = 113;
const getKey = (obj,val) => Object.keys(obj).find(key => obj[key] === val); // For getting tags by tagid


let model, emodel;
(async function() {
    model = await tf.loadLayersModel('http://deepdivision.net/NERjs/tfjs_models/ner/model.json');
    let outputs_ = [model.output, model.getLayer("attention_vector").output];
    emodel = tf.model({inputs: model.input, outputs: outputs_});
    $('.loading-model').remove();
    $('.form').removeClass("hide");
})();


function word_preprocessor(word) {
  word = word.replace(/[-|.|,|\?|\!]+/g, '');
  word = word.replace(/\d+/g, '1');
  word = word.toLowerCase();
  if (word != '') {
    return word;
  } else {
    return '.'
  }
};

function make_sequences(words_array) {
  let sequence = Array();
  words_array.slice(0, MAX_SEQUENCE_LENGTH).forEach(function(word) {
    word = word_preprocessor(word);
    let id = words_vocab[word];
    if (id == undefined) {
      sequence.push(words_vocab['<UNK>']);
    } else {
      sequence.push(id);
    }  
  });

  // pad sequence
  if (sequence.length < MAX_SEQUENCE_LENGTH) {
    let pad_array = Array(MAX_SEQUENCE_LENGTH - sequence.length);
    pad_array.fill(words_vocab['<UNK>']);
    sequence = sequence.concat(pad_array);
  }

  return sequence;
};

async function make_predict() {
    $(".main-result").html("");
    $('.attention-bar').html("");
    $(".tags-result").html("<h5>Tags review</h5><table class='table table-sm table-bordered tags-review'></table>");

    let words = $('#input_text').val().split(' ');
    let sequence = make_sequences(words);
    let tensor = tf.tensor1d(sequence, dtype='int32')
      .expandDims(0);
    let [predictions, attention_probs] = await emodel.predict(tensor);
    attention_probs = await attention_probs.data();
    
    predictions = await predictions.argMax(-1).data();
    let predictions_tags = Array();
    predictions.forEach(function(tagid) {
      predictions_tags.push(getKey(tags_vocab, tagid));
    });

    words.forEach(function(word, index) {
      let current_word = word;
      if (['B-ORG', 'I-ORG'].includes(predictions_tags[index])) {
        current_word += " <span class='badge badge-primary'>"+predictions_tags[index]+"</span>";
      };
      if (['B-PER', 'I-PER'].includes(predictions_tags[index])) {
        current_word += " <span class='badge badge-info'>"+predictions_tags[index]+"</span>";
      };
      if (['B-LOC', 'I-LOC'].includes(predictions_tags[index])) {
        current_word += " <span class='badge badge-success'>"+predictions_tags[index]+"</span>";
      };
      if (['B-MISC', 'I-MISC'].includes(predictions_tags[index])) {
        current_word += " <span class='badge badge-warning'>"+predictions_tags[index]+"</span>";
      };
      $(".main-result").append(current_word+' ');
    });

    let x = Array(), y = Array();
    words.forEach(function(word, index) {
      x.push(word);
      y.push(attention_probs[index]);
    });
    let plot_data = {x: x, y: y, type: 'bar'};
    Plotly.newPlot('attention_bar', [plot_data], {height: 300});
    $('.attention-bar').prepend("<h5>Attention</h5>");

    $('.tags-review').append("<tr id='tags-words'><th>normalized word</th></tr>");
    words.forEach(function(word) {
      $('#tags-words').append("<td>"+word_preprocessor(word)+"</td>");
    });

    $('.tags-review').append("<tr id='tags-sen'><th>word_id <i>(0 - PAD, 1 - UNK)</i></th></tr>");
    sequence.slice(0, words.length).forEach(function(tok) {
      $('#tags-sen').append("<td>"+tok+"</td>");
    });

    $('.tags-review').append("<tr id='tags-ner'><th>token</th></tr>");
    predictions_tags.slice(0, words.length).forEach(function(tok) {
      $('#tags-ner').append("<td>"+tok+"</td>");
    });

    // for (let index = 0; index < words.length; index++) {
    //   var tag_col = words[index]+"<hr>"+predictions_tags[index];
    //   $(".tags-review").append("<div class='col tag'>"+tag_col+"</div>");    
    //   console.log(tag_col);
    // }
};

$("#get_ner_button").click(make_predict);
$('#input_text').keypress(function (e) {
    if (e.which == 13) {
      make_predict();
    }
  });
