$(document).ready(function (e) {
    $('#finderImage').click(function (e) {
        var posX = $(this).offset().left,
            posY =  $(this).height() + $(this).offset().top;
        var sourcePosX = e.pageX - posX,
            sourcePosY = posY - e.pageY;
        alert(sourcePosX + ' , ' + sourcePosY);

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

$(document).ready(function (e) {
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
        context.strokeStyle = "#000000";
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