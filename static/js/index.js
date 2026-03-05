window.addEventListener('DOMContentLoaded', function () {
  var burger = document.querySelector('.navbar-burger');
  if (burger) {
    var targetId = burger.dataset.target;
    var target = targetId ? document.getElementById(targetId) : null;
    burger.addEventListener('click', function () {
      burger.classList.toggle('is-active');
      if (target) {
        target.classList.toggle('is-active');
      }
    });
  }

	  var VISER_PLACEHOLDER_TEXT = 'Choose a scene above';
	  var VISER_READY_TEXT = 'Click and move me';

	  function getViewerBanner(viewer) {
	    if (!viewer || !viewer.parentElement) {
	      return null;
	    }
	    return viewer.parentElement.querySelector('.viser-banner');
	  }

	  function getViewerHint(viewer) {
	    if (!viewer) {
	      return null;
	    }
	    var container = viewer.closest('.interactive-card--viewer, .dataset-block--viewer');
	    if (!container) {
	      return null;
	    }
	    return container.querySelector('[data-viser-hint]');
	  }

	  function setViewerHintVisible(viewer, visible) {
	    var hint = getViewerHint(viewer);
	    if (!hint) {
	      return;
	    }
	    hint.classList.toggle('is-hidden', !visible);
	  }

	  function setViewerPlaybackDock(viewer, enabled) {
	    if (!viewer || !viewer.parentElement) {
	      return;
	    }
	    viewer.parentElement.classList.toggle('pw-viser-with-playback', !!enabled);
	  }

	  function setViewerBanner(banner, isPlaceholder) {
	    if (!banner) {
	      return;
	    }
	    if (isPlaceholder) {
	      banner.textContent = VISER_PLACEHOLDER_TEXT;
	      banner.classList.add('is-placeholder');
	      banner.classList.remove('is-hidden');
	      return;
	    }
	    banner.textContent = VISER_READY_TEXT;
	    banner.classList.remove('is-placeholder');
	    banner.classList.add('is-hidden');
	  }


  var DEFAULT_CAMERA = {
    position: '1.00,0.00,1.00',
    lookAt: '0.00,0.00,0.00',
    up: '0.000,0.000,1.000'
  };

  function mergeCamera(camera, fallback) {
    var base = fallback || DEFAULT_CAMERA;
    return {
      position: camera && camera.position ? camera.position : base.position,
      lookAt: camera && camera.lookAt ? camera.lookAt : base.lookAt,
      up: camera && camera.up ? camera.up : base.up
    };
  }

	  function buildViewerSrc(base, filename, camera, fallback, dockPlayback) {
	    var merged = mergeCamera(camera, fallback);
	    var rawPath = '../../' + base + '/' + filename;
	    var encoded = encodeURI(rawPath).replace(/\+/g, '%2B');
	    return 'static/viser-client/index.html?playbackPath=' + encoded +
	      '&initialCameraPosition=' + encodeURIComponent(merged.position) +
	      '&initialCameraLookAt=' + encodeURIComponent(merged.lookAt) +
	      '&initialCameraUp=' + encodeURIComponent(merged.up) +
	      (dockPlayback ? '&pwDockPlayback=1' : '');
	  }

  initInteractiveSection();
  initDatasetSection();

  function initInteractiveSection() {
    // Shared viewers (lazy init)
    var predFrame = document.getElementById('interactive-pred');
    var gtFrame = document.getElementById('interactive-gt');
    if (predFrame) { predFrame.removeAttribute('src'); predFrame.dataset.base = ''; }
    if (gtFrame) { gtFrame.removeAttribute('src'); gtFrame.dataset.base = ''; }
	    var predBanner = getViewerBanner(predFrame);
	    var gtBanner = getViewerBanner(gtFrame);
	    setViewerBanner(predBanner, true);
	    setViewerBanner(gtBanner, true);
	    setViewerHintVisible(predFrame, false);
	    setViewerHintVisible(gtFrame, false);
	    setViewerPlaybackDock(predFrame, false);
	    setViewerPlaybackDock(gtFrame, false);

    function getInteractiveCamera(button, target) {
      if (!button) {
        return DEFAULT_CAMERA;
      }
      var suffix = target ? '-' + target : '';
      var position = button.getAttribute('data-camera-position' + suffix) || button.getAttribute('data-camera-position');
      var lookAt = button.getAttribute('data-camera-lookat' + suffix) || button.getAttribute('data-camera-lookat');
      var up = button.getAttribute('data-camera-up' + suffix) || button.getAttribute('data-camera-up');
      return {
        position: position || DEFAULT_CAMERA.position,
        lookAt: lookAt || DEFAULT_CAMERA.lookAt,
        up: up || DEFAULT_CAMERA.up
      };
    }

    function initOneCarousel(opts) {
      var container = opts.container;
      var thumbRow = container.querySelector(opts.thumbRowSelector);
      var thumbs = Array.from(container.querySelectorAll(opts.thumbRowSelector + ' .interactive-thumb'));
      var dotsContainer = container.querySelector(opts.dotsSelector);
      var inputsCard = document.getElementById(opts.inputsCardId);
      var imageRoles = opts.imageRoles; // e.g., ['rgb0','depth0','rgb1','depth1'] or with rgb2/depth2
      var imageMap = {};
      imageRoles.forEach(function (r) { imageMap[r] = inputsCard ? inputsCard.querySelector('[data-role="' + r + '"]') : null; });

      var interactiveDots = [];
      if (dotsContainer) {
        dotsContainer.innerHTML = '';
        interactiveDots = thumbs.map(function (_, idx) {
          var dot = document.createElement('button');
          dot.type = 'button';
          dot.className = 'carousel-dot' + (idx === 0 && opts.autoActivate ? ' is-active' : '');
          dot.setAttribute('aria-label', 'Show interactive scene ' + (idx + 1));
          dot.addEventListener('click', function () { api.activate(idx); });
          dotsContainer.appendChild(dot);
          return dot;
        });
      }

      // Magnifier bound to this inputs grid
      var interactiveMagnifier = (function () {
        var zoom = 2.5;
        var lensSize = 180;
        var lensByRole = {};
        function ensureLens(role) {
          if (!lensByRole[role]) {
            var lens = document.createElement('div');
            lens.className = 'dataset-magnifier-lens is-hidden';
            document.body.appendChild(lens);
            lensByRole[role] = lens;
          }
          return lensByRole[role];
        }
        function hideAll() {
          Object.keys(lensByRole).forEach(function (k) { var l = lensByRole[k]; if (l) l.classList.add('is-hidden'); });
        }
        function computeDisplayPos(img, clientX, clientY) {
          var rect = img.getBoundingClientRect();
          var x = clientX - rect.left; var y = clientY - rect.top;
          x = Math.max(0, Math.min(rect.width, x));
          y = Math.max(0, Math.min(rect.height, y));
          return { x: x, y: y, rect: rect };
        }
        function applyLens(lens, img, dispPos) {
          var bgSize = (dispPos.rect.width * zoom) + 'px ' + (dispPos.rect.height * zoom) + 'px';
          var bgPosX = -(dispPos.x * zoom - lensSize / 2);
          var bgPosY = -(dispPos.y * zoom - lensSize / 2);
          lens.style.backgroundImage = 'url("' + (img.currentSrc || img.src) + '")';
          lens.style.backgroundSize = bgSize;
          lens.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
        }
        function placeLensNearPoint(lens, viewportX, viewportY) {
          var offset = 12; var left = viewportX + offset; var top = viewportY - lensSize / 2;
          if (left + lensSize > window.innerWidth - 8) { left = viewportX - lensSize - offset; }
          top = Math.max(8, Math.min(window.innerHeight - lensSize - 8, top));
          lens.style.left = Math.round(left) + 'px'; lens.style.top = Math.round(top) + 'px';
        }
        function onMove(evt, role) {
          var img = imageMap[role]; if (!img) return;
          var lens = ensureLens(role);
          var disp = computeDisplayPos(img, evt.clientX, evt.clientY);
          applyLens(lens, img, disp);
          placeLensNearPoint(lens, disp.rect.left + disp.x, disp.rect.top + disp.y);
          lens.classList.remove('is-hidden');
        }
        function onLeave(role) { var lens = ensureLens(role); if (lens) lens.classList.add('is-hidden'); }
        function bind(img, role) {
          if (!img) return;
          img.addEventListener('mouseenter', function (e) { onMove(e, role); });
          img.addEventListener('mousemove', function (e) { onMove(e, role); });
          img.addEventListener('mouseleave', function () { onLeave(role); });
          window.addEventListener('scroll', function () { onLeave(role); }, { passive: true });
          window.addEventListener('resize', function () { onLeave(role); });
        }
        function init() { imageRoles.forEach(function (role) { bind(imageMap[role], role); }); }
        return { init: init, hideAll: hideAll };
      })();

      function updateImages(base, label) {
        function set(role, path, alt) { if (imageMap[role]) { imageMap[role].src = base + '/' + path; imageMap[role].alt = (label || '') + ' ' + alt; } }
        set('rgb0', 'cameras-rgb/cam0.png', 'RGB cam0');
        set('depth0', 'cameras-depth/cam0.png', 'depth cam0');
        set('rgb1', 'cameras-rgb/cam1.png', 'RGB cam1');
        set('depth1', 'cameras-depth/cam1.png', 'depth cam1');
        if (imageMap.rgb2) set('rgb2', 'cameras-rgb/cam2.png', 'RGB cam2');
        if (imageMap.depth2) set('depth2', 'cameras-depth/cam2.png', 'depth cam2');
      }

      var activeIdx = -1;
      function activate(idx) {
        if (idx < 0 || idx >= thumbs.length) return;
        activeIdx = idx;
        thumbs.forEach(function (btn, i) { btn.classList.toggle('is-active', i === idx); btn.setAttribute('aria-pressed', i === idx ? 'true' : 'false'); });
        interactiveDots.forEach(function (dot, i) { dot.classList.toggle('is-active', i === idx); });
        var button = thumbs[idx]; var base = button.getAttribute('data-base'); var label = button.getAttribute('data-label') || '';
        var cameraPred = getInteractiveCamera(button, 'pred');
        var cameraGt = getInteractiveCamera(button, 'gt');
        if (!base) return;
        if (interactiveMagnifier && interactiveMagnifier.hideAll) interactiveMagnifier.hideAll();
        updateImages(base, label);
	        if (predFrame && predFrame.dataset.base !== base) { predFrame.src = buildViewerSrc(base, 'scene-pred.viser', cameraPred, null, true); predFrame.dataset.base = base; }
	        if (gtFrame && gtFrame.dataset.base !== base) { gtFrame.src = buildViewerSrc(base, 'scene-gt.viser', cameraGt, null, true); gtFrame.dataset.base = base; }
	        setViewerBanner(predBanner, false);
	        setViewerBanner(gtBanner, false);
	        setViewerHintVisible(predFrame, true);
	        setViewerHintVisible(gtFrame, true);
	        setViewerPlaybackDock(predFrame, true);
	        setViewerPlaybackDock(gtFrame, true);
	      }

      // Bind thumb clicks
      thumbs.forEach(function (btn, idx) { btn.addEventListener('click', function () { activate(idx); }); });
      // Bind arrows scoped to this carousel
      container.querySelectorAll('.interactive-arrow').forEach(function (arrow) {
        arrow.addEventListener('click', function (event) {
          event.preventDefault();
          var direction = arrow.getAttribute('data-direction') === 'prev' ? -1 : 1;
          var next = activeIdx < 0 ? (direction === -1 ? thumbs.length - 1 : 0) : (activeIdx + direction + thumbs.length) % thumbs.length;
          activate(next);
          if (thumbs[next]) {
            thumbs[next].focus();
            if (thumbRow) {
              var rowRect = thumbRow.getBoundingClientRect();
              var thumbRect = thumbs[next].getBoundingClientRect();
              if (thumbRect.left < rowRect.left || thumbRect.right > rowRect.right) {
                thumbs[next].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
              }
            }
          }
        });
      });

      // Initialize magnifier for this inputs grid
      if (interactiveMagnifier) interactiveMagnifier.init();

      var api = { activate: activate, getActiveIndex: function () { return activeIdx; } };
      if (opts.autoActivate) activate(0);
      return api;
    }

    // Initialize both carousels (stacked, always visible)
    var droidApi = initOneCarousel({
      container: document.getElementById('interactive-row-droid'),
      thumbRowSelector: '#interactive-thumb-row-droid',
      dotsSelector: '#interactive-dots-droid',
      inputsCardId: 'interactive-inputs-droid',
      imageRoles: ['rgb0','depth0','rgb1','depth1'],
      autoActivate: false
    });
    var b1kApi = initOneCarousel({
      container: document.getElementById('interactive-row-b1k'),
      thumbRowSelector: '#interactive-thumb-row-b1k',
      dotsSelector: '#interactive-dots-b1k',
      inputsCardId: 'interactive-inputs-b1k',
      imageRoles: ['rgb0','rgb1','rgb2','depth0','depth1','depth2'],
      autoActivate: false
    });

    // Show inputs card matching the last interacted carousel
    function showInputs(which) {
      var isDroid = which === 'droid';
      document.getElementById('interactive-inputs-droid').classList.toggle('is-hidden', !isDroid);
      document.getElementById('interactive-inputs-b1k').classList.toggle('is-hidden', isDroid);
    }
    showInputs('droid');

    // Keep B1K tile size roughly matching DROID tile size
    function syncB1KGridWidth() {
      var droidGrid = document.querySelector('#interactive-inputs-droid .interactive-image-grid');
      var b1kGrid = document.querySelector('#interactive-inputs-b1k .interactive-image-grid--b1k');
      if (!droidGrid || !b1kGrid) return;
      var firstDroidImg = droidGrid.querySelector('img');
      if (!firstDroidImg) return;
      var tileW = firstDroidImg.clientWidth;
      if (!tileW) return;
      var gapPx = 0;
      try {
        var cs = window.getComputedStyle(droidGrid);
        var gap = cs.getPropertyValue('gap') || cs.getPropertyValue('grid-gap') || '0px';
        gapPx = parseFloat(gap);
        if (isNaN(gapPx)) gapPx = 0;
      } catch (e) {}
      var desired = Math.round((tileW * 3) + (gapPx * 2));
      // Set a max-width so it fits on small screens while matching on larger ones
      b1kGrid.style.maxWidth = desired + 'px';
      b1kGrid.style.width = '100%';
      b1kGrid.style.marginLeft = 'auto';
      b1kGrid.style.marginRight = 'auto';
    }
    // Run after layout settles
    setTimeout(syncB1KGridWidth, 0);
    window.addEventListener('resize', syncB1KGridWidth);
    // Wire up listeners so clicking in a row also flips inputs card
    var droidRow = document.getElementById('interactive-row-droid');
    var b1kRow = document.getElementById('interactive-row-b1k');
    if (droidRow) droidRow.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('.interactive-thumb')) {
        showInputs('droid');
      }
    });
    if (b1kRow) b1kRow.addEventListener('click', function (e) {
      if (e.target.closest && e.target.closest('.interactive-thumb')) {
        // ensure first b1k sample is activated on first interaction
        if (b1kApi && b1kApi.getActiveIndex && b1kApi.getActiveIndex() === 0) {
          // activate(0) may already be called; safe to call again
        }
        showInputs('b1k');
      }
    });
  }

  function initDatasetSection() {
    var section = document.getElementById('dataset');
    if (!section) {
      return;
    }
    var datasetThumbs = Array.from(section.querySelectorAll('.dataset-thumb'));
    if (!datasetThumbs.length) {
      return;
    }

    var activeIdx = -1; // no default sample selected
    var thumbRow = section.querySelector('#dataset-thumb-row');
    var dotsContainer = section.querySelector('#dataset-dots');
    var datasetDots = [];
    if (dotsContainer) {
      dotsContainer.innerHTML = '';
      datasetDots = datasetThumbs.map(function (_, idx) {
        var dot = document.createElement('button');
        dot.type = 'button';
        dot.className = 'carousel-dot' + (idx === activeIdx ? ' is-active' : '');
        dot.setAttribute('aria-label', 'Show dataset sample ' + (idx + 1));
        dot.addEventListener('click', function () {
          activateDatasetSample(idx);
        });
        dotsContainer.appendChild(dot);
        return dot;
      });
    }
	    var oursViewer = document.getElementById('dataset-viewer-ours');
	    var originalViewer = document.getElementById('dataset-viewer-original');
	    var oursBanner = getViewerBanner(oursViewer);
	    var originalBanner = getViewerBanner(originalViewer);
	    setViewerBanner(oursBanner, true);
	    setViewerBanner(originalBanner, true);
	    setViewerHintVisible(oursViewer, false);
	    setViewerHintVisible(originalViewer, false);
    var imageMap = {
      'ours-rgb-0': section.querySelector('[data-dataset-role="ours-rgb-0"]'),
      'ours-depth-0': section.querySelector('[data-dataset-role="ours-depth-0"]'),
      'ours-rgb-1': section.querySelector('[data-dataset-role="ours-rgb-1"]'),
      'ours-depth-1': section.querySelector('[data-dataset-role="ours-depth-1"]'),
      'original-rgb-0': section.querySelector('[data-dataset-role="original-rgb-0"]'),
      'original-depth-0': section.querySelector('[data-dataset-role="original-depth-0"]'),
      'original-rgb-1': section.querySelector('[data-dataset-role="original-rgb-1"]'),
      'original-depth-1': section.querySelector('[data-dataset-role="original-depth-1"]')
    };

    // --- Synchronized magnifier setup for Calibrated RGB + Depth ---
    var magnifier = (function () {
      var zoom = 2.5; // magnification factor
      var lensSize = 180; // must match CSS width/height
      var lensByKey = {}; // key: 'rgb-0' | 'depth-0' | 'rgb-1' | 'depth-1'

      function ensureLenses(key) {
        if (!lensByKey[key]) {
          var lensOurs = document.createElement('div');
          lensOurs.className = 'dataset-magnifier-lens is-hidden';
          var lensOrig = document.createElement('div');
          lensOrig.className = 'dataset-magnifier-lens is-hidden';
          document.body.appendChild(lensOurs);
          document.body.appendChild(lensOrig);
          lensByKey[key] = { ours: lensOurs, original: lensOrig };
        }
        return lensByKey[key];
      }

      function hideAll() {
        Object.keys(lensByKey).forEach(function (k) {
          var pair = lensByKey[k];
          if (pair && pair.ours) pair.ours.classList.add('is-hidden');
          if (pair && pair.original) pair.original.classList.add('is-hidden');
        });
      }

      function roleToKey(role) {
        // role is like 'ours-rgb-0' -> returns { key: 'rgb-0', side: 'ours' }
        var parts = role.split('-');
        if (parts.length < 3) return { key: role, side: '' };
        var side = parts[0];
        var key = parts.slice(1).join('-');
        return { key: key, side: side };
      }

      function pairedRole(role) {
        var info = roleToKey(role);
        var otherSide = info.side === 'ours' ? 'original' : 'ours';
        return otherSide + '-' + info.key;
      }

      function computeDisplayPos(img, clientX, clientY) {
        var rect = img.getBoundingClientRect();
        var x = clientX - rect.left; // in displayed px
        var y = clientY - rect.top;
        // clamp inside image bounds
        x = Math.max(0, Math.min(rect.width, x));
        y = Math.max(0, Math.min(rect.height, y));
        return { x: x, y: y, rect: rect };
      }

      function applyLens(lens, img, dispPos) {
        // background sizing based on displayed size keeps mapping simple
        var bgSize = (dispPos.rect.width * zoom) + 'px ' + (dispPos.rect.height * zoom) + 'px';
        var bgPosX = -(dispPos.x * zoom - lensSize / 2);
        var bgPosY = -(dispPos.y * zoom - lensSize / 2);
        lens.style.backgroundImage = 'url("' + (img.currentSrc || img.src) + '")';
        lens.style.backgroundSize = bgSize;
        lens.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
      }

      function placeLensNearPoint(lens, viewportX, viewportY) {
        var offset = 12; // px
        var left = viewportX + offset;
        var top = viewportY - lensSize / 2;
        // flip horizontally if overflowing viewport
        if (left + lensSize > window.innerWidth - 8) {
          left = viewportX - lensSize - offset;
        }
        // clamp vertically
        top = Math.max(8, Math.min(window.innerHeight - lensSize - 8, top));
        lens.style.left = Math.round(left) + 'px';
        lens.style.top = Math.round(top) + 'px';
      }

      function onMove(evt, role) {
        var srcImg = imageMap[role];
        var pairRole = pairedRole(role);
        var dstImg = imageMap[pairRole];
        if (!srcImg || !dstImg) return;

        var info = roleToKey(role);
        var lenses = ensureLenses(info.key);
        var srcLens = info.side === 'ours' ? lenses.ours : lenses.original;
        var dstLens = info.side === 'ours' ? lenses.original : lenses.ours;

        // positions in displayed coordinates for both images
        var srcDisp = computeDisplayPos(srcImg, evt.clientX, evt.clientY);
        // map proportionally onto the paired image
        var ratioX = srcDisp.x / (srcDisp.rect.width || 1);
        var ratioY = srcDisp.y / (srcDisp.rect.height || 1);
        var dstRect = dstImg.getBoundingClientRect();
        var dstX = ratioX * dstRect.width;
        var dstY = ratioY * dstRect.height;

        // update the lenses' backgrounds
        applyLens(srcLens, srcImg, srcDisp);
        applyLens(dstLens, dstImg, { x: dstX, y: dstY, rect: dstRect });

        // place lenses near their respective points
        placeLensNearPoint(srcLens, srcDisp.rect.left + srcDisp.x, srcDisp.rect.top + srcDisp.y);
        placeLensNearPoint(dstLens, dstRect.left + dstX, dstRect.top + dstY);

        srcLens.classList.remove('is-hidden');
        dstLens.classList.remove('is-hidden');
      }

      function onLeave(role) {
        var info = roleToKey(role);
        var lenses = ensureLenses(info.key);
        if (lenses.ours) lenses.ours.classList.add('is-hidden');
        if (lenses.original) lenses.original.classList.add('is-hidden');
      }

      function bind(img, role) {
        if (!img) return;
        img.addEventListener('mouseenter', function (e) { onMove(e, role); });
        img.addEventListener('mousemove', function (e) { onMove(e, role); });
        img.addEventListener('mouseleave', function () { onLeave(role); });
        // also hide on scroll to avoid stray lenses
        window.addEventListener('scroll', function () { onLeave(role); }, { passive: true });
        window.addEventListener('resize', function () { onLeave(role); });
      }

      function init() {
        ['rgb-0', 'depth-0', 'rgb-1', 'depth-1'].forEach(function (key) {
          bind(imageMap['ours-' + key], 'ours-' + key);
          bind(imageMap['original-' + key], 'original-' + key);
        });
      }

      return { init: init, hideAll: hideAll };
    })();

    function getDatasetCamera(button) {
      if (!button) {
        return DEFAULT_CAMERA;
      }
      return {
        position: button.getAttribute('data-camera-position') || DEFAULT_CAMERA.position,
        lookAt: button.getAttribute('data-camera-lookat') || DEFAULT_CAMERA.lookAt,
        up: button.getAttribute('data-camera-up') || DEFAULT_CAMERA.up
      };
    }

    function updateDatasetImages(base, label) {
      var labelText = label || 'Dataset sample';
      var mappings = [
        ['ours-rgb-0', base + '/fs-refined/rgb_robot/cam0.png', 'ours RGB cam0'],
        ['ours-depth-0', base + '/fs-refined/depth/cam0.png', 'ours depth cam0'],
        ['ours-rgb-1', base + '/fs-refined/rgb_robot/cam1.png', 'ours RGB cam1'],
        ['ours-depth-1', base + '/fs-refined/depth/cam1.png', 'ours depth cam1'],
        ['original-rgb-0', base + '/raw-tri/rgb_robot/cam0.png', 'original RGB cam0'],
        ['original-depth-0', base + '/raw-tri/depth/cam0.png', 'original depth cam0'],
        ['original-rgb-1', base + '/raw-tri/rgb_robot/cam1.png', 'original RGB cam1'],
        ['original-depth-1', base + '/raw-tri/depth/cam1.png', 'original depth cam1']
      ];
      mappings.forEach(function (entry) {
        var role = entry[0];
        var src = entry[1];
        var altSuffix = entry[2];
        var img = imageMap[role];
        if (img) {
          img.src = src;
          img.alt = labelText + ' ' + altSuffix;
        }
      });
    }

    function activateDatasetSample(index) {
      if (index < 0 || index >= datasetThumbs.length) {
        return;
      }
      activeIdx = index;
      datasetThumbs.forEach(function (btn, idx) {
        btn.classList.toggle('is-active', idx === index);
        btn.setAttribute('aria-pressed', idx === index ? 'true' : 'false');
      });
      datasetDots.forEach(function (dot, idx) {
        dot.classList.toggle('is-active', idx === index);
      });
      var button = datasetThumbs[index];
      var base = button.getAttribute('data-base');
      var label = button.getAttribute('data-label') || '';
      var camera = getDatasetCamera(button);
      if (!base) {
        return;
      }
      updateDatasetImages(base, label);
      // hide magnifiers when switching samples
      if (magnifier && magnifier.hideAll) { magnifier.hideAll(); }
	      if (oursViewer && oursViewer.dataset.base !== base) {
	        oursViewer.src = buildViewerSrc(base, 'scene-fs-refined.viser', camera);
	        oursViewer.dataset.base = base;
	      }
	      if (originalViewer && originalViewer.dataset.base !== base) {
	        originalViewer.src = buildViewerSrc(base, 'scene-raw-tri.viser', camera);
	        originalViewer.dataset.base = base;
	      }
	      setViewerBanner(oursBanner, false);
	      setViewerBanner(originalBanner, false);
	      setViewerHintVisible(oursViewer, true);
	      setViewerHintVisible(originalViewer, true);
	    }

    datasetThumbs.forEach(function (btn, idx) {
      btn.addEventListener('click', function () {
        activateDatasetSample(idx);
      });
    });

    section.querySelectorAll('.dataset-arrow').forEach(function (arrow) {
      arrow.addEventListener('click', function (event) {
        event.preventDefault();
        var direction = arrow.getAttribute('data-direction') === 'prev' ? -1 : 1;
        var next = activeIdx < 0 ? (direction === -1 ? datasetThumbs.length - 1 : 0) : (activeIdx + direction + datasetThumbs.length) % datasetThumbs.length;
        activateDatasetSample(next);
        if (datasetThumbs[next]) {
          datasetThumbs[next].focus();
        }
        if (thumbRow && datasetThumbs[next]) {
          var nextThumb = datasetThumbs[next];
          var rowRect = thumbRow.getBoundingClientRect();
          var thumbRect = nextThumb.getBoundingClientRect();
          if (thumbRect.left < rowRect.left || thumbRect.right > rowRect.right) {
            nextThumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
          }
        }
      });
    });

    if (oursViewer) {
      oursViewer.removeAttribute('src');
      oursViewer.dataset.base = '';
    }
    if (originalViewer) {
      originalViewer.removeAttribute('src');
      originalViewer.dataset.base = '';
    }

    // Initialize synchronized magnifiers for calibrated inputs
    if (magnifier) {
      magnifier.init();
    }
  }
});
