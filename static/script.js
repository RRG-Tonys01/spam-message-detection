document.addEventListener('DOMContentLoaded', function() {
    var envelope = document.getElementById('envelope');
    var btn_check = document.getElementById('check');
    var btn_reset = document.getElementById('reset');
    var result = document.getElementById('result');
    const form = document.getElementById('myForm');

    envelope.addEventListener('click', function() {
        open();
    });

    btn_check.addEventListener('click', function() {
        sendDataToFlask();
    });

    btn_reset.addEventListener('click', function() {
        close();
    });

    function open() {
        envelope.classList.add('open');
        envelope.classList.remove('close');
    }

    function close() {
        envelope.classList.add('close');
        envelope.classList.remove('open');
        result.innerHTML = "";
        form.reset();
    }

    function sendDataToFlask(){
        const formData = new FormData(form);

        fetch('/message', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Response from Flask:', data);
            result.innerHTML = data['message'];
            open();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
