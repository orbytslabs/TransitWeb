$(document).ready(function (e) {
    $('#finderImageCanvas').click(function (e) {
        var posX = $(this).offset().left,
            posY =  $(this).height() + $(this).offset().top;
        var sourcePosX = e.pageX - posX,
            sourcePosY = posY - e.pageY;

        $.ajax({
          type: "POST",
          url: "/source-selection/create-photometry-source",
          data: { 
            "sourcePosX": sourcePosX, 
            "sourcePosY": sourcePosY,
            csrfmiddlewaretoken: $("input[name='csrfmiddlewaretoken']").val() 
          },
          success: function(response){
            console.log('Success');
          },
        });
    });
});

$('#content-finder-image').ready(function (e) {

    $('input:checkbox').click(function() {
        $('input:checkbox').not(this).prop('checked', false);
    });

    var canvas = document.getElementById("finderImageCanvas");
    var context = canvas.getContext('2d');

    function createImageOnCanvas(imageId) {
        canvas.style.display = "block";
        document.getElementById("images").style.overflowY = "hidden";
        var img = new Image(300, 300);
        img.src = document.getElementById(imageId).src;
        context.drawImage(img, (0), (0)); //onload....
    }

    function draw(e) {
        var pos = getMousePos(canvas, e);
        posx = pos.x;
        posy = pos.y;
        context.fillStyle = "transparent";
        if ($('#target').is(":checked"))
        {
          context.strokeStyle = "#000000"; 
        }
        else if ($('#comparison').is(":checked")) {
          context.strokeStyle = "#FF0000";
        }
        context.lineWidth = 2;
        context.beginPath();
        context.arc(posx, posy, 8, 0, 2*Math.PI);
        context.fill();
        context.stroke();
    }

    function getMousePos(canvas, evt) {
        var rect = canvas.getBoundingClientRect();
        return {
          x: evt.clientX - rect.left,
          y: evt.clientY - rect.top
        };
    }

    window.draw = draw;
});